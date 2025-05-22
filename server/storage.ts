import { 
  User, 
  InsertUser, 
  VideoAnalysis, 
  VideoAnalysisResult
} from "@shared/schema";
import session from "express-session";
import createMemoryStore from "memorystore";
import { randomBytes, createHash } from "crypto";

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
}

export const storage = new MemStorage();
