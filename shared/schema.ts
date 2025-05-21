import { pgTable, text, serial, integer, boolean, timestamp, json } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User table schema
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  role: text("role").default("user").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

// Video analysis table schema
export const videoAnalyses = pgTable("video_analyses", {
  id: text("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  fileName: text("file_name").notNull(),
  fileSize: integer("file_size"),
  uploadDate: timestamp("upload_date").defaultNow().notNull(),
  isDeepfake: boolean("is_deepfake"),
  confidence: integer("confidence"),
  processingTime: integer("processing_time"),
  analysisData: json("analysis_data"),
});

export const insertVideoAnalysisSchema = createInsertSchema(videoAnalyses).omit({
  id: true,
});

// Type definitions
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertVideoAnalysis = z.infer<typeof insertVideoAnalysisSchema>;
export type VideoAnalysis = typeof videoAnalyses.$inferSelect;

// Video analysis result type for the frontend
export type VideoAnalysisResult = {
  id: string;
  fileName: string;
  userId: number;
  uploadDate: string;
  fileSize?: number;
  analysis: {
    isDeepfake: boolean;
    confidence: number;
    processingTime: number;
    issues?: { type: string; text: string }[];
    findings?: {
      title: string;
      icon: string;
      severity: string;
      timespan: string;
      description: string;
    }[];
    timeline?: {
      position: number;
      tooltip: string;
      type: string;
    }[];
  };
};
