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
      const pythonScript = path.join(__dirname, 'simple_detector.py');
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
  
  // Create a simple video endpoint for demo purposes
  app.get("/uploads/:videoId", async (req, res) => {
    try {
      // For demonstration purposes, we'll return a simple MP4 video stream
      // In a production app, we would retrieve the specific video file for this ID from storage
      res.setHeader('Content-Type', 'video/mp4');
      
      // Check if video exists in our analysis database
      const videoId = req.params.videoId;
      const videoAnalysis = await storage.getVideoAnalysis(videoId, req.user?.id);
      
      if (!videoAnalysis) {
        return res.status(404).send("Video not found");
      }
      
      // In a real implementation, we would stream the actual video file
      // For demo, we embed a video in base64 format - in production we'd use real storage
      const fs = require('fs');
      const path = require('path');
      
      // Use our demo video for all uploads in this prototype
      const videoPath = path.join(process.cwd(), 'public', 'sample.mp4');
      
      // If we have a demo video, stream it
      if (fs.existsSync(videoPath)) {
        fs.createReadStream(videoPath).pipe(res);
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

      // Generate a unique video ID
      const videoId = `${Date.now()}-${req.user.id}-${Math.floor(Math.random() * 1000)}`;
      
      // Save video file temporarily for analysis
      const tempVideoPath = path.join(__dirname, '..', 'uploads', `${videoId}.${req.file.originalname.split('.').pop()}`);
      
      // Ensure uploads directory exists
      const uploadsDir = path.dirname(tempVideoPath);
      if (!fs.existsSync(uploadsDir)) {
        fs.mkdirSync(uploadsDir, { recursive: true });
      }
      
      // Write the uploaded video to temporary file
      fs.writeFileSync(tempVideoPath, req.file.buffer);
      
      // Run PyTorch deepfake analysis
      let analysisData;
      try {
        console.log("Starting deepfake analysis for video:", tempVideoPath);
        analysisData = await runDeepfakeAnalysis(tempVideoPath);
        console.log("Analysis completed. Results:", JSON.stringify(analysisData, null, 2));
        
        // Clean up temporary file after analysis
        fs.unlinkSync(tempVideoPath);
        
        if (analysisData.error) {
          throw new Error(analysisData.error);
        }
      } catch (error) {
        // Clean up temporary file on error
        if (fs.existsSync(tempVideoPath)) {
          fs.unlinkSync(tempVideoPath);
        }
        console.error("Deepfake analysis failed:", error);
        return res.status(500).json({ 
          message: "Failed to analyze video", 
          error: error.message 
        });
      }
      
      // Extract analysis results from PyTorch model
      const isDeepfake = analysisData.isDeepfake;
      const confidence = Math.round(analysisData.confidence * 100);
      const processingTime = analysisData.processingTime || 1;
      const findings = analysisData.findings || [];
      const timeline = analysisData.timeline || [];
      const issues = analysisData.issues || [];
      
      // Construct the full analysis result
      const analysisResult = {
        id: videoId,
        fileName: req.file.originalname,
        userId: req.user.id,
        uploadDate: new Date().toISOString(),
        fileSize: Math.floor(fileSize / 1024), // Convert to KB
        analysis: {
          isDeepfake,
          confidence,
          processingTime,
          issues,
          findings,
          timeline
        }
      };

      await storage.saveVideoAnalysis(videoId, req.user.id, analysisResult);

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
        processingTimes
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

  const httpServer = createServer(app);
  return httpServer;
}
