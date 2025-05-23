import { 
  User, 
  InsertUser, 
  VideoAnalysis, 
  VideoAnalysisResult
} from "@shared/schema";
import session from "express-session";
import createMemoryStore from "memorystore";
import { randomBytes, createHash } from "crypto";
import fs from "fs";

const MemoryStore = createMemoryStore(session);

// Define the system log type
export type SystemLog = {
  id: string;
  timestamp: string;
  type: 'error' | 'warning' | 'info';
  source: string;
  message: string;
  details: string;
};

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  getAllUsers(): Promise<User[]>;
  getUserCount(): Promise<number>;
  
  // Video analysis operations
  saveVideoAnalysis(id: string, userId: number, analysis: any): Promise<void>;
  getVideoAnalysis(id: string, userId: number): Promise<VideoAnalysisResult | undefined>;
  getUserVideoAnalyses(userId: number): Promise<VideoAnalysisResult[]>;
  getVideoCount(): Promise<number>;
  getDeepfakeCount(): Promise<number>;
  
  // System logs operations
  getSystemLogs(): Promise<SystemLog[]>;
  addSystemLog(log: Omit<SystemLog, 'id' | 'timestamp'>): Promise<SystemLog>;
  clearSystemLogs(): Promise<void>;
  
  // Cache management operations
  clearVideoCache(): Promise<{ deletedCount: number, totalSize: number }>;
  getLastCacheClearTime(): Date | null;
  getCacheInfo(): Promise<{ videoCount: number, estimatedSize: number, lastCleared: Date | null }>;
  
  // Session store
  sessionStore: session.Store;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private videoAnalyses: Map<string, VideoAnalysisResult>;
  private systemLogs: Map<string, SystemLog>;
  public sessionStore: session.Store;
  private currentId: number;
  private logId: number;
  private lastCacheCleared: Date | null = null;

  constructor() {
    this.users = new Map();
    this.videoAnalyses = new Map();
    this.systemLogs = new Map();
    this.currentId = 1;
    this.logId = 1;
    this.sessionStore = new MemoryStore({
      checkPeriod: 86400000, // 24 hours
    });
    
    // Create the admin user with simpler hashing
    const admin: User = {
      id: this.currentId++,
      username: "admin",
      password: createHash("sha256").update("admin123").digest("hex"),
      role: "admin",
      createdAt: new Date()
    };
    
    this.users.set(admin.id, admin);
    console.log("Created default admin user");
    
    // Add initial system logs
    const initialLog: Omit<SystemLog, 'id' | 'timestamp'> = {
      type: "info",
      source: "system", 
      message: "System initialized",
      details: "DeepGuard AI system started successfully"
    };
    this.addSystemLog(initialLog);
    
    const adminLog: Omit<SystemLog, 'id' | 'timestamp'> = {
      type: "info",
      source: "auth",
      message: "Default admin user created",
      details: "Username: admin"
    };
    this.addSystemLog(adminLog);
  }

  // User methods
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentId++;
    const role = insertUser.username.includes("admin") ? "admin" : "user"; // Simple role assignment for demo
    const createdAt = new Date();
    
    const user: User = { 
      ...insertUser, 
      id,
      role,
      createdAt
    };
    
    this.users.set(id, user);
    return user;
  }

  async getAllUsers(): Promise<User[]> {
    return Array.from(this.users.values());
  }

  async getUserCount(): Promise<number> {
    return this.users.size;
  }

  // Video analysis methods
  async saveVideoAnalysis(id: string, userId: number, analysis: any): Promise<void> {
    this.videoAnalyses.set(id, analysis);
  }

  async getVideoAnalysis(id: string, userId: number): Promise<VideoAnalysisResult | undefined> {
    const analysis = this.videoAnalyses.get(id);
    
    // Only return the analysis if it belongs to the requesting user
    if (analysis && analysis.userId === userId) {
      return analysis;
    }
    
    return undefined;
  }

  async getUserVideoAnalyses(userId: number): Promise<VideoAnalysisResult[]> {
    return Array.from(this.videoAnalyses.values()).filter(
      (analysis) => analysis.userId === userId
    );
  }

  async getVideoCount(): Promise<number> {
    return this.videoAnalyses.size;
  }

  async getDeepfakeCount(): Promise<number> {
    return Array.from(this.videoAnalyses.values()).filter(
      (analysis) => analysis.analysis.isDeepfake
    ).length;
  }
  
  // System logs methods
  async getSystemLogs(): Promise<SystemLog[]> {
    return Array.from(this.systemLogs.values()).sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }
  
  async addSystemLog(log: Omit<SystemLog, 'id' | 'timestamp'>): Promise<SystemLog> {
    const id = `log_${this.logId++}`;
    const timestamp = new Date().toISOString();
    
    const newLog: SystemLog = {
      id,
      timestamp,
      ...log
    };
    
    this.systemLogs.set(id, newLog);
    return newLog;
  }
  
  async clearVideoCache(): Promise<{ deletedCount: number, totalSize: number }> {
    // Calculate approximate size before clearing (15MB per video is a rough estimate)
    const videoCount = this.videoAnalyses.size;
    const estimatedSize = videoCount * 15; // Size in MB
    
    // Get all video IDs for logging
    const videoIds = Array.from(this.videoAnalyses.keys());
    
    // Attempt to delete the physical video files
    let deletedFiles = 0;
    let totalSizeBytes = 0;
    
    try {
      // Get the uploads directory path
      const uploadsDir = './uploads';
      
      // Check if directory exists
      if (fs.existsSync(uploadsDir)) {
        // Read all files in the directory
        const files = fs.readdirSync(uploadsDir);
        
        // Delete each video file
        for (const file of files) {
          // Only delete mp4 files
          if (file.endsWith('.mp4')) {
            const filePath = `${uploadsDir}/${file}`;
            
            try {
              // Get file size before deleting
              const stats = fs.statSync(filePath);
              totalSizeBytes += stats.size;
              
              // Delete the file
              fs.unlinkSync(filePath);
              deletedFiles++;
              
              console.log(`Deleted video file: ${filePath}`);
            } catch (fileError) {
              console.error(`Failed to delete file ${filePath}:`, fileError);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error deleting video files:', error);
    }
    
    // Clear the video analyses from memory
    this.videoAnalyses.clear();
    
    // Log this operation
    await this.addSystemLog({
      type: 'info',
      source: 'storage',
      message: `Cleared video cache: ${deletedFiles} files deleted`,
      details: `Freed approximately ${(totalSizeBytes / (1024 * 1024)).toFixed(2)} MB of disk space`
    });
    
    // Update the last cleared timestamp
    this.lastCacheCleared = new Date();
    
    // Log the cache clearing operation
    await this.addSystemLog({
      type: 'info',
      source: 'admin',
      message: `Cache cleared by admin`,
      details: `Cleared ${videoCount} videos (approx. ${estimatedSize}MB)`
    });
    
    // Look for actual video files in the uploads directory and delete them
    try {
      const uploadsDir = './uploads';
      if (fs.existsSync(uploadsDir)) {
        const files = fs.readdirSync(uploadsDir);
        let filesDeleted = 0;
        
        for (const file of files) {
          // Only delete video files
          if (file.endsWith('.mp4') || file.endsWith('.avi') || file.endsWith('.mov')) {
            fs.unlinkSync(`${uploadsDir}/${file}`);
            filesDeleted++;
          }
        }
        
        // Add extra log if physical files were deleted
        if (filesDeleted > 0) {
          await this.addSystemLog({
            type: 'info',
            source: 'admin',
            message: `Physical video files deleted`,
            details: `Removed ${filesDeleted} video files from uploads directory`
          });
        }
      }
    } catch (error) {
      console.error('Error deleting physical video files:', error);
      // We don't throw here, as clearing the in-memory cache is still considered successful
    }
    
    return {
      deletedCount: videoCount,
      totalSize: estimatedSize
    };
  }
  
  getLastCacheClearTime(): Date | null {
    return this.lastCacheCleared;
  }
  
  async getCacheInfo(): Promise<{ videoCount: number, estimatedSize: number, lastCleared: Date | null }> {
    const videoCount = this.videoAnalyses.size;
    const estimatedSize = videoCount * 15; // Rough estimate: 15MB per video
    
    return {
      videoCount,
      estimatedSize,
      lastCleared: this.lastCacheCleared
    };
  }
}

export const storage = new MemStorage();
