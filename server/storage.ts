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
import Database from "better-sqlite3";

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

// SQLite Storage Implementation
export class SQLiteStorage implements IStorage {
  private db: Database.Database;
  public sessionStore: session.Store;
  private lastCacheCleared: Date | null = null;

  constructor() {
    // Initialize SQLite database
    this.db = new Database('./data.db');
    
    // Create tables
    this.initializeTables();
    
    // Create session store
    this.sessionStore = new MemoryStore({
      checkPeriod: 86400000 // Prune expired entries every 24h
    });
    
    // Create default admin user if it doesn't exist
    this.initializeDefaultAdmin();
  }

  private initializeTables() {
    // Users table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Video analyses table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS video_analyses (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        file_name TEXT NOT NULL,
        upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        file_size INTEGER,
        analysis TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
      )
    `);

    // System logs table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS system_logs (
        id TEXT PRIMARY KEY,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        type TEXT NOT NULL,
        source TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT
      )
    `);
  }

  private initializeDefaultAdmin() {
    const existingAdmin = this.db.prepare("SELECT * FROM users WHERE username = ?").get("admin");
    
    if (!existingAdmin) {
      const hashedPassword = createHash('sha256').update('admin').digest('hex');
      this.db.prepare("INSERT INTO users (username, password) VALUES (?, ?)").run("admin", hashedPassword);
      console.log("Created default admin user");
    }
  }

  async getUser(id: number): Promise<User | undefined> {
    const user = this.db.prepare("SELECT * FROM users WHERE id = ?").get(id) as User | undefined;
    return user;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const user = this.db.prepare("SELECT * FROM users WHERE username = ?").get(username) as User | undefined;
    return user;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const { username, password } = insertUser;
    const hashedPassword = createHash('sha256').update(password).digest('hex');
    
    const result = this.db.prepare("INSERT INTO users (username, password) VALUES (?, ?) RETURNING *").get(username, hashedPassword) as User;
    return result;
  }

  async getAllUsers(): Promise<User[]> {
    const users = this.db.prepare("SELECT * FROM users").all() as User[];
    return users;
  }

  async getUserCount(): Promise<number> {
    const result = this.db.prepare("SELECT COUNT(*) as count FROM users").get() as { count: number };
    return result.count;
  }

  async saveVideoAnalysis(id: string, userId: number, analysis: any): Promise<void> {
    const analysisJson = JSON.stringify(analysis);
    this.db.prepare("INSERT OR REPLACE INTO video_analyses (id, user_id, file_name, file_size, analysis) VALUES (?, ?, ?, ?, ?)")
      .run(id, userId, analysis.fileName || 'unknown', analysis.fileSize || 0, analysisJson);
  }

  async getVideoAnalysis(id: string, userId: number): Promise<VideoAnalysisResult | undefined> {
    const row = this.db.prepare("SELECT * FROM video_analyses WHERE id = ? AND user_id = ?").get(id, userId) as any;
    
    if (!row) return undefined;
    
    const analysis = JSON.parse(row.analysis);
    return {
      id: row.id,
      fileName: row.file_name,
      userId: row.user_id,
      uploadDate: row.upload_date,
      fileSize: row.file_size,
      analysis
    };
  }

  async getUserVideoAnalyses(userId: number): Promise<VideoAnalysisResult[]> {
    const rows = this.db.prepare("SELECT * FROM video_analyses WHERE user_id = ? ORDER BY upload_date DESC").all(userId) as any[];
    
    return rows.map(row => ({
      id: row.id,
      fileName: row.file_name,
      userId: row.user_id,
      uploadDate: row.upload_date,
      fileSize: row.file_size,
      analysis: JSON.parse(row.analysis)
    }));
  }

  async getVideoCount(): Promise<number> {
    const result = this.db.prepare("SELECT COUNT(*) as count FROM video_analyses").get() as { count: number };
    return result.count;
  }

  async getDeepfakeCount(): Promise<number> {
    const rows = this.db.prepare("SELECT analysis FROM video_analyses").all() as { analysis: string }[];
    
    let deepfakeCount = 0;
    for (const row of rows) {
      try {
        const analysis = JSON.parse(row.analysis);
        if (analysis.isDeepfake) {
          deepfakeCount++;
        }
      } catch (error) {
        // Skip invalid JSON
      }
    }
    
    return deepfakeCount;
  }

  async getSystemLogs(): Promise<SystemLog[]> {
    const logs = this.db.prepare("SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT 100").all() as SystemLog[];
    return logs;
  }

  async addSystemLog(log: Omit<SystemLog, 'id' | 'timestamp'>): Promise<SystemLog> {
    const id = `log_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const timestamp = new Date().toISOString();
    
    const newLog: SystemLog = {
      id,
      timestamp,
      ...log
    };
    
    this.db.prepare("INSERT INTO system_logs (id, timestamp, type, source, message, details) VALUES (?, ?, ?, ?, ?, ?)")
      .run(id, timestamp, log.type, log.source, log.message, log.details || '');
    
    return newLog;
  }

  async clearVideoCache(): Promise<{ deletedCount: number, totalSize: number }> {
    const videoCount = await this.getVideoCount();
    const estimatedSize = videoCount * 15; // Size in MB
    
    // Clear video analyses from database
    this.db.prepare("DELETE FROM video_analyses").run();
    
    // Log this operation
    await this.addSystemLog({
      type: 'info',
      source: 'storage',
      message: `Cleared video cache: ${videoCount} records deleted`,
      details: `Freed approximately ${estimatedSize} MB of estimated disk space`
    });
    
    this.lastCacheCleared = new Date();
    
    return {
      deletedCount: videoCount,
      totalSize: estimatedSize
    };
  }
  
  getLastCacheClearTime(): Date | null {
    return this.lastCacheCleared;
  }
  
  async getCacheInfo(): Promise<{ videoCount: number, estimatedSize: number, lastCleared: Date | null }> {
    const videoCount = await this.getVideoCount();
    const estimatedSize = videoCount * 15; // Rough estimate: 15MB per video
    
    return {
      videoCount,
      estimatedSize,
      lastCleared: this.lastCacheCleared
    };
  }
}

export const storage = new SQLiteStorage();
