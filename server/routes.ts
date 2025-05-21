import type { Express, Request } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { setupAuth } from "./auth";
import multer from "multer";

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

  // Videos API
  app.post("/api/videos/upload", upload.single("video"), async (req: any, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }

      if (!req.file) {
        return res.status(400).json({ message: "No video file provided" });
      }

      // In a real implementation, we would:
      // 1. Save the video file
      // 2. Process it with Gemini API for deepfake detection
      // 3. Store the analysis results

      // Simulated analysis for demo purposes
      const videoId = Date.now().toString();
      const analysisResult = {
        id: videoId,
        fileName: req.file.originalname,
        userId: req.user.id,
        uploadDate: new Date().toISOString(),
        fileSize: req.file.size,
        analysis: {
          isDeepfake: Math.random() > 0.5,
          confidence: Math.floor(Math.random() * 30) + 70, // 70-100% confidence
          processingTime: Math.floor(Math.random() * 5) + 1, // 1-5 seconds
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
  app.get("/api/admin/stats", async (req, res) => {
    try {
      if (!req.isAuthenticated()) {
        return res.status(401).json({ message: "Authentication required" });
      }
      
      // In a real app, we would check proper admin status
      if (!req.user.username.includes("admin")) {
        return res.status(403).json({ message: "Admin access required" });
      }
      
      // Simplified stats for the demo
      const stats = {
        totalUsers: await storage.getUserCount(),
        videosAnalyzed: await storage.getVideoCount(),
        deepfakesDetected: await storage.getDeepfakeCount(),
        systemHealth: 98.7
      };
      
      res.json(stats);
    } catch (error) {
      res.status(500).json({ 
        message: "Failed to fetch admin stats",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

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

  const httpServer = createServer(app);
  return httpServer;
}
