import { 
  User, 
  InsertUser, 
  VideoAnalysis, 
  VideoAnalysisResult
} from "@shared/schema";
import session from "express-session";
import createMemoryStore from "memorystore";

const MemoryStore = createMemoryStore(session);

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
  
  // Session store
  sessionStore: session.Store;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private videoAnalyses: Map<string, VideoAnalysisResult>;
  public sessionStore: session.Store;
  private currentId: number;

  constructor() {
    this.users = new Map();
    this.videoAnalyses = new Map();
    this.currentId = 1;
    this.sessionStore = new MemoryStore({
      checkPeriod: 86400000, // 24 hours
    });
    
    // Add a default admin user
    this.createUser({
      username: "admin",
      password: "admin123" // In a real app, this would be hashed
    }).then(admin => {
      console.log("Created default admin user");
    });
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
}

export const storage = new MemStorage();
