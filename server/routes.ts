import type { Express, Request } from "express";
import express from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { setupAuth } from "./auth";
import multer from "multer";
import { VideoAnalysisResult } from "../shared/schema";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import { processDeepfakeQuery, getDeepfakeTips, analyzeTimelineMarker } from "./gemini-service";

// We'll use any for our types here to bypass TypeScript errors
// In a production app, we would define proper types
type AuthenticatedRequest = Request & { 
  user: any;
  isAuthenticated(): boolean;
};

export async function registerRoutes(app: Express): Promise<Server> {
  // Set up authentication routes
  setupAuth(app);

  // Configure file storage for video uploads
  const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
      fileSize: 100 * 1024 * 1024, // 100MB limit
    },
    fileFilter: (req: any, file: any, cb: any) => {
      // Accept only video files
      if (file.mimetype.startsWith("video/")) {
        cb(null, true);
      } else {
        cb(new Error("Only video files are allowed"));
      }
    },
  });

  // Serve static files from public directory
  app.use('/public', express.static('public'));

  // Function to run deepfake analysis using Python model
  async function runDeepfakeAnalysis(videoPath: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const pythonScript = path.join(process.cwd(), 'server', 'simple_detector.py');
      const pythonProcess = spawn('python3', [pythonScript, videoPath]);
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (error) {
            reject(new Error(`Failed to parse analysis result: ${error}`));
          }
        } else {
          reject(new Error(`Python script failed with code ${code}: ${stderr}`));
        }
      });
      
      pythonProcess.on('error', (error) => {
        reject(new Error(`Failed to start Python process: ${error.message}`));
      });
    });
  }
  
  // Video serving endpoint
  app.get("/uploads/:videoId", async (req, res) => {
    try {
      // Check if video exists in our analysis database
      const videoId = req.params.videoId;
      const userId = req.user?.id || 0; // Provide default value if user ID is undefined
      const videoAnalysis = await storage.getVideoAnalysis(videoId, userId);
      
      if (!videoAnalysis) {
        return res.status(404).send("Video not found");
      }
      
      // Set proper content type for video response
      res.setHeader('Content-Type', 'video/mp4');
      
      // Check if we have the video stored in the viewing directory (this is where we save uploaded videos for playback)
      const viewingVideoPath = path.join(process.cwd(), 'uploads', 'viewing', `${videoId}.mp4`);
      
      // If the viewing video exists, serve that
      if (fs.existsSync(viewingVideoPath)) {
        fs.createReadStream(viewingVideoPath).pipe(res);
        return;
      }
      
      // Fallback: check old upload directory (for backward compatibility)
      const uploadedVideoPath = path.join(process.cwd(), 'uploads', `${videoId}.mp4`);
      
      if (fs.existsSync(uploadedVideoPath)) {
        fs.createReadStream(uploadedVideoPath).pipe(res);
        return;
      }
      
      // Fallback to sample video if no actual uploaded video is found
      const samplePath = path.join(process.cwd(), 'public', 'sample.mp4');
      if (fs.existsSync(samplePath)) {
        fs.createReadStream(samplePath).pipe(res);
      } else {
        // Fallback to a simple message if no video is available
        res.status(404).send("Video file not available");
      }
    } catch (error) {
      console.error("Error serving video file:", error);
      res.status(500).send("Error processing video request");
    }
  });

  // Videos API
  app.post("/api/videos/upload", upload.single("video"), async (req: any, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }

      if (!req.file) {
        return res.status(400).json({ message: "No video file provided" });
      }

      // Generate a unique video ID with more randomness to prevent duplicate analysis
      const videoId = `${Date.now()}-${req.user.id}-${Math.floor(Math.random() * 10000)}`;
      
      // Save video file temporarily for analysis
      const tempVideoPath = path.join(process.cwd(), 'uploads', `${videoId}.${req.file.originalname.split('.').pop()}`);
      
      // Ensure uploads directory exists
      const uploadsDir = path.dirname(tempVideoPath);
      if (!fs.existsSync(uploadsDir)) {
        fs.mkdirSync(uploadsDir, { recursive: true });
      }
      
      // Write the uploaded video to temporary file for analysis
      fs.writeFileSync(tempVideoPath, req.file.buffer);
      
      // Also save a permanent copy for viewing (different location to prevent bias)
      const viewingVideoPath = path.join(process.cwd(), 'uploads', 'viewing', `${videoId}.mp4`);
      const viewingDir = path.dirname(viewingVideoPath);
      console.log("Creating viewing directory:", viewingDir);
      if (!fs.existsSync(viewingDir)) {
        fs.mkdirSync(viewingDir, { recursive: true });
        console.log("Viewing directory created successfully");
      }
      console.log("Saving viewing copy to:", viewingVideoPath);
      fs.writeFileSync(viewingVideoPath, req.file.buffer);
      console.log("Viewing copy saved successfully, file size:", fs.statSync(viewingVideoPath).size, "bytes");
      
      // Run PyTorch deepfake analysis
      let analysisData;
      try {
        console.log("Starting deepfake analysis for video:", tempVideoPath);
        // Force a new analysis by adding a unique marker to ensure it doesn't reuse previous results
        analysisData = await runDeepfakeAnalysis(tempVideoPath);
        console.log("Analysis completed. Results:", JSON.stringify(analysisData, null, 2));
        
        // Delete the temporary analysis file after analysis to prevent bias
        try {
          fs.unlinkSync(tempVideoPath);
          console.log(`Temporary analysis file ${tempVideoPath} has been deleted to prevent bias`);
          console.log(`Video preserved for viewing at: ${viewingVideoPath}`);
        } catch (deleteError) {
          console.error(`Failed to delete temporary analysis file: ${deleteError}`);
        }
        
        if (analysisData.error) {
          throw new Error(analysisData.error);
        }
      } catch (error) {
        // Delete the temporary analysis file even if analysis fails (but keep viewing copy)
        try {
          fs.unlinkSync(tempVideoPath);
          console.log(`Temporary analysis file deleted after failed analysis`);
        } catch (deleteError) {
          console.error(`Failed to delete temporary analysis file: ${deleteError}`);
        }
        
        console.error("Deepfake analysis failed:", error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        return res.status(500).json({ 
          message: "Failed to analyze video", 
          error: errorMessage 
        });
      }
      
      // Extract analysis results from PyTorch model
      const isDeepfake = analysisData.isDeepfake;
      const confidence = analysisData.confidence; // Keep the raw confidence value (0-1 range)
      const processingTime = analysisData.processingTime || 1;
      const findings = analysisData.findings || [];
      const timeline = analysisData.timeline || [];
      const issues = analysisData.issues || [];
      
      // Construct the full analysis result
      const analysisResult: VideoAnalysisResult = {
        id: videoId,
        fileName: req.file.originalname,
        userId: req.user.id,
        uploadDate: new Date().toISOString(),
        fileSize: req.file.size ? Math.floor(req.file.size / 1024) : undefined, // Convert to KB if size exists
        analysis: {
          isDeepfake,
          confidence,
          processingTime,
          issues,
          findings,
          timeline
        }
      };

      // Save the analysis to storage
      await storage.saveVideoAnalysis(videoId, req.user.id, analysisResult);
      
      // Log the analysis for monitoring
      await storage.addSystemLog({
        type: "info",
        source: "video-analysis",
        message: `New video analyzed: ${req.file.originalname}`,
        details: `Result: ${isDeepfake ? 'Deepfake' : 'Authentic'}, Confidence: ${(confidence * 100).toFixed(1)}%`
      });

      res.status(200).json({
        success: true,
        videoId,
        message: "Video uploaded and analysis started",
      });
    } catch (error) {
      console.error("Video upload error:", error);
      res.status(500).json({ 
        message: "Failed to process video upload",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Get video analysis by ID
  app.get("/api/videos/:id", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }

      const analysis = await storage.getVideoAnalysis(req.params.id, req.user.id);
      
      if (!analysis) {
        return res.status(404).json({ message: "Analysis not found" });
      }
      
      res.json(analysis);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch video analysis",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Get all video analyses for the authenticated user
  app.get("/api/videos", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }

      const analyses = await storage.getUserVideoAnalyses(req.user.id);
      res.json(analyses);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch user video analyses",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });
  
  // Clear all video analyses and delete video files
  app.post("/api/videos/clear-all", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }
      
      // Admin check
      if (!req.user.username.includes("admin")) {
        return res.status(403).json({ message: "Admin access required" });
      }
      
      // Clear all videos from storage and disk
      const result = await storage.clearVideoCache();
      
      // Return success message with details
      res.json({
        success: true,
        message: "All video files and analyses have been cleared",
        deletedCount: result.deletedCount,
        freedSpace: `${(result.totalSize / (1024 * 1024)).toFixed(2)} MB`
      });
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to clear video cache",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Admin routes - For simplicity in this demo, we'll check if the username contains "admin"
  async function adminStatsHandler(req: Request, res: any) {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }
      
      // In a real app, we would check proper admin status
      if (!req.user.username.includes("admin")) {
        return res.status(403).json({ message: "Admin access required" });
      }
      
      // Get basic stats from database
      const totalUsers = await storage.getUserCount();
      const videosAnalyzed = await storage.getVideoCount();
      const deepfakesDetected = await storage.getDeepfakeCount();
      
      // Get all videos to generate data for charts
      const users = await storage.getAllUsers();
      const allVideos: VideoAnalysisResult[] = [];
      
      // Collect all video analyses
      for (const user of users) {
        const userVideos = await storage.getUserVideoAnalyses(user.id);
        allVideos.push(...userVideos);
      }
      
      // Sort videos by date
      allVideos.sort((a, b) => new Date(a.uploadDate).getTime() - new Date(b.uploadDate).getTime());
      
      // Generate daily uploads data based on actual upload dates
      const videosByDate = new Map();
      const now = new Date();
      
      // Initialize the last 7 days
      for (let i = 6; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);
        const formattedDate = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        videosByDate.set(formattedDate, 0);
      }
      
      // Count videos uploaded on each day
      allVideos.forEach(video => {
        const uploadDate = new Date(video.uploadDate);
        // Only count videos from the last 7 days
        if (now.getTime() - uploadDate.getTime() <= 7 * 24 * 60 * 60 * 1000) {
          const formattedDate = uploadDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
          if (videosByDate.has(formattedDate)) {
            videosByDate.set(formattedDate, videosByDate.get(formattedDate) + 1);
          }
        }
      });
      
      // Convert to array format for the frontend
      const dailyUploads = Array.from(videosByDate.entries()).map(([date, count]) => ({
        date,
        count
      }));
      
      // Calculate detection rates by day
      const detectionRates = dailyUploads.map(item => {
        const date = item.date;
        const dateVideos = allVideos.filter(video => {
          const uploadDate = new Date(video.uploadDate);
          return uploadDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) === date;
        });
        
        const totalForDay = dateVideos.length;
        const deepfakesForDay = dateVideos.filter(video => video.analysis.isDeepfake).length;
        const rate = totalForDay > 0 ? (deepfakesForDay / totalForDay) * 100 : 0;
        
        return {
          date,
          rate: parseFloat(rate.toFixed(1))
        };
      });
      
      // Calculate deepfake types from analysis data
      const deepfakeTypes = new Map();
      deepfakeTypes.set("Facial Manipulation", 0);
      deepfakeTypes.set("Voice Synthesis", 0);
      deepfakeTypes.set("Body Movements", 0);
      deepfakeTypes.set("Background Alterations", 0);
      
      // Count different types of deepfakes based on findings
      allVideos.forEach(video => {
        if (video.analysis.isDeepfake && video.analysis.findings) {
          video.analysis.findings.forEach((finding: { 
            title: string; 
            icon: string; 
            severity: string; 
            timespan: string; 
            description: string; 
          }) => {
            if (finding.title.includes("Facial")) {
              deepfakeTypes.set("Facial Manipulation", deepfakeTypes.get("Facial Manipulation") + 1);
            } else if (finding.title.includes("Audio")) {
              deepfakeTypes.set("Voice Synthesis", deepfakeTypes.get("Voice Synthesis") + 1);
            } else if (finding.title.includes("movement")) {
              deepfakeTypes.set("Body Movements", deepfakeTypes.get("Body Movements") + 1);
            } else if (finding.title.includes("Light")) {
              deepfakeTypes.set("Background Alterations", deepfakeTypes.get("Background Alterations") + 1);
            }
          });
        }
      });
      
      // Convert to array format for the frontend
      const detectionTypes = Array.from(deepfakeTypes.entries()).map(([name, value]) => ({
        name,
        value: value || 0 // Ensure we don't have undefined values
      }));
      
      // Calculate user growth over time
      const usersByWeek = new Map();
      const now2 = new Date();
      
      // Initialize the last 5 weeks
      for (let i = 4; i >= 0; i--) {
        const date = new Date(now2);
        date.setDate(date.getDate() - (i * 7));
        const formattedDate = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        usersByWeek.set(formattedDate, 0);
      }
      
      // Count users registered by week
      users.forEach(user => {
        const creationDate = new Date(user.createdAt);
        // Only consider users from the last 5 weeks
        if (now2.getTime() - creationDate.getTime() <= 5 * 7 * 24 * 60 * 60 * 1000) {
          // Find the week this user falls into
          for (let i = 4; i >= 0; i--) {
            const weekStart = new Date(now2);
            weekStart.setDate(weekStart.getDate() - (i * 7));
            const weekEnd = new Date(weekStart);
            weekEnd.setDate(weekEnd.getDate() + 7);
            
            if (creationDate >= weekStart && creationDate < weekEnd) {
              const formattedDate = weekStart.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
              if (usersByWeek.has(formattedDate)) {
                usersByWeek.set(formattedDate, usersByWeek.get(formattedDate) + 1);
              }
              break;
            }
          }
        }
      });
      
      // Convert to array format for the frontend
      const userGrowth = Array.from(usersByWeek.entries()).map(([date, users]) => ({
        date,
        users
      }));
      
      // Calculate processing time distributions
      const processingTimesMap = new Map();
      processingTimesMap.set("<30s", 0);
      processingTimesMap.set("30s-1m", 0);
      processingTimesMap.set("1m-2m", 0);
      processingTimesMap.set("2m-5m", 0);
      processingTimesMap.set(">5m", 0);
      
      // Categorize videos by processing time
      allVideos.forEach(video => {
        const processingTime = video.analysis.processingTime || 0;
        
        if (processingTime < 30) {
          processingTimesMap.set("<30s", processingTimesMap.get("<30s") + 1);
        } else if (processingTime < 60) {
          processingTimesMap.set("30s-1m", processingTimesMap.get("30s-1m") + 1);
        } else if (processingTime < 120) {
          processingTimesMap.set("1m-2m", processingTimesMap.get("1m-2m") + 1);
        } else if (processingTime < 300) {
          processingTimesMap.set("2m-5m", processingTimesMap.get("2m-5m") + 1);
        } else {
          processingTimesMap.set(">5m", processingTimesMap.get(">5m") + 1);
        }
      });
      
      // Convert to array format for the frontend
      const processingTimes = Array.from(processingTimesMap.entries()).map(([timeRange, count]) => ({
        timeRange,
        count
      }));
      
      // System health is a derived metric from actual system performance
      // Here we'll calculate it based on successful video analysis ratio
      const systemHealth = allVideos.length > 0 ? 
        (allVideos.filter(v => v.analysis && v.analysis.isDeepfake !== undefined).length / allVideos.length) * 100 : 
        100;
      
      // Calculate classification breakdown
      const authenticCount = allVideos.filter(v => !v.analysis.isDeepfake && v.analysis.confidence < 0.7).length;
      const deepfakeCount = allVideos.filter(v => v.analysis.isDeepfake).length;
      const moderateCount = allVideos.filter(v => !v.analysis.isDeepfake && v.analysis.confidence >= 0.7).length;
      
      const classificationBreakdown = [
        { name: "Authentic", value: authenticCount, color: "#00ff88" },
        { name: "Deepfake", value: deepfakeCount, color: "#ff3366" },
        { name: "Moderate/Suspicious", value: moderateCount, color: "#ffaa00" }
      ];

      // Construct complete analytics response
      const stats = {
        summary: {
          totalUsers,
          videoCount: videosAnalyzed,
          deepfakesDetected,
          systemHealth: parseFloat(systemHealth.toFixed(1))
        },
        dailyUploads,
        detectionRates,
        detectionTypes,
        userGrowth,
        processingTimes,
        classificationBreakdown
      };
      
      res.json(stats);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch admin stats",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  }

  // Register both endpoints to support the transition
  app.get("/api/admin/stats", adminStatsHandler);
  app.get("/api/admin/analytics", adminStatsHandler);
  
  app.get("/api/admin/users", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }
      
      // In a real app, we would check proper admin status
      if (!req.user.username.includes("admin")) {
        return res.status(403).json({ message: "Admin access required" });
      }
      
      const users = await storage.getAllUsers();
      res.json(users);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch users",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });
  
  // System logs endpoint - retrieves logs from the database
  app.get("/api/admin/logs", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }
      
      // In a real app, we would check proper admin status
      if (!req.user.username.includes("admin")) {
        return res.status(403).json({ message: "Admin access required" });
      }
      
      // Get logs from storage (database)
      const logs = await storage.getSystemLogs();
      
      // Add a log entry for this logs access
      await storage.addSystemLog({
        type: "info",
        source: "admin",
        message: "System logs accessed",
        details: `Admin user: ${req.user.username}`
      });
      
      res.json(logs);
    } catch (error) {
      // Log the error
      await storage.addSystemLog({
        type: "error",
        source: "api",
        message: "Failed to fetch system logs",
        details: error instanceof Error ? error.message : "Unknown error"
      });
      
      res.status(500).json({ 
        message: "Failed to fetch system logs",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Chatbot API - Process deepfake-related questions
  app.post("/api/chatbot/query", express.json(), async (req, res) => {
    try {
      const { message } = req.body;
      
      if (!message) {
        return res.status(400).json({ error: "Message is required" });
      }
      
      // Process the query using Gemini AI
      const response = await processDeepfakeQuery(message);
      
      res.json({ response });
    } catch (error) {
      console.error("Error processing chatbot query:", error);
      res.status(500).json({ 
        error: "Failed to process query",
        message: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });
  
  // Get deepfake detection tips
  app.get("/api/chatbot/tips", async (req, res) => {
    try {
      const tips = await getDeepfakeTips();
      res.json({ tips });
    } catch (error) {
      console.error("Error getting deepfake tips:", error);
      res.status(500).json({ 
        error: "Failed to get deepfake tips",
        message: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });
  
  // Analyze timeline marker with detailed explanation
  app.post("/api/analyze-timeline-marker", async (req, res) => {
    try {
      const { markerType, markerTooltip, timestamp } = req.body;
      
      if (!markerType || !markerTooltip || !timestamp) {
        return res.status(400).json({ 
          error: "Missing required fields",
          message: "markerType, markerTooltip, and timestamp are required" 
        });
      }
      
      // Validate marker type
      if (!['normal', 'warning', 'danger'].includes(markerType)) {
        return res.status(400).json({ 
          error: "Invalid marker type",
          message: "markerType must be 'normal', 'warning', or 'danger'" 
        });
      }
      
      const analysis = await analyzeTimelineMarker(
        markerType as 'normal' | 'warning' | 'danger',
        markerTooltip,
        timestamp
      );
      
      res.json({ analysis });
    } catch (error) {
      console.error("Error analyzing timeline marker:", error);
      res.status(500).json({ 
        error: "Failed to analyze timeline marker",
        message: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
