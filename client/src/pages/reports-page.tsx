import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function ReportsPage() {
  const { user } = useAuth();
  const [timeRange, setTimeRange] = useState("last7days");
  
  // Simulated data for reports
  const summaryStats = {
    totalVideos: 27,
    deepfakesDetected: 12,
    detectionRate: 44.4,
    averageConfidence: 89.2,
  };
  
  const recentAnalyses = [
    {
      id: "1",
      fileName: "conference_speech.mp4",
      uploadDate: "2025-05-20T14:30:00Z",
      isDeepfake: true,
      confidence: 93.7,
    },
    {
      id: "2",
      fileName: "interview_clip.mp4",
      uploadDate: "2025-05-19T09:15:00Z",
      isDeepfake: false,
      confidence: 88.2,
    },
    {
      id: "3",
      fileName: "social_media_video.mp4",
      uploadDate: "2025-05-18T16:45:00Z",
      isDeepfake: true,
      confidence: 91.5,
    },
    {
      id: "4",
      fileName: "news_segment.mp4",
      uploadDate: "2025-05-17T11:20:00Z",
      isDeepfake: false,
      confidence: 94.3,
    },
    {
      id: "5",
      fileName: "political_address.mp4",
      uploadDate: "2025-05-16T13:10:00Z",
      isDeepfake: true,
      confidence: 87.8,
    },
  ];
  
  const deepfakeTypes = [
    { type: "Facial Manipulation", count: 7, percentage: 58.3 },
    { type: "Voice Synthesis", count: 3, percentage: 25.0 },
    { type: "Body Movements", count: 1, percentage: 8.3 },
    { type: "Background Alteration", count: 1, percentage: 8.3 },
  ];

  // Format date helper
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    }).format(date);
  };

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar isAdmin={user?.role === "admin"} />
      
      <div className="flex-1 ml-20 md:ml-64 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
            <h1 className="text-3xl font-bold">Analysis Reports</h1>
            
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Time Range:</span>
              <Select value={timeRange} onValueChange={setTimeRange}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Select range" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="last7days">Last 7 Days</SelectItem>
                  <SelectItem value="last30days">Last 30 Days</SelectItem>
                  <SelectItem value="last90days">Last 90 Days</SelectItem>
                  <SelectItem value="lastYear">Last Year</SelectItem>
                  <SelectItem value="allTime">All Time</SelectItem>
                </SelectContent>
              </Select>
              
              <Button variant="outline">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
                Export
              </Button>
            </div>
          </div>
          
          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="glass rounded-xl p-6">
              <div className="flex items-center gap-4">
                <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="m10 8 6 4-6 4Z"/></svg>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Total Videos</p>
                  <h3 className="text-2xl font-bold">{summaryStats.totalVideos}</h3>
                </div>
              </div>
            </div>
            
            <div className="glass rounded-xl p-6">
              <div className="flex items-center gap-4">
                <div className="h-12 w-12 rounded-full bg-destructive/10 flex items-center justify-center text-destructive">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Deepfakes Detected</p>
                  <h3 className="text-2xl font-bold">{summaryStats.deepfakesDetected}</h3>
                </div>
              </div>
            </div>
            
            <div className="glass rounded-xl p-6">
              <div className="flex items-center gap-4">
                <div className="h-12 w-12 rounded-full bg-secondary/10 flex items-center justify-center text-secondary">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Detection Rate</p>
                  <h3 className="text-2xl font-bold">{summaryStats.detectionRate}%</h3>
                </div>
              </div>
            </div>
            
            <div className="glass rounded-xl p-6">
              <div className="flex items-center gap-4">
                <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21 2-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0 3 3L22 7l-3-3m-3.5 3.5L19 4"/></svg>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg. Confidence</p>
                  <h3 className="text-2xl font-bold">{summaryStats.averageConfidence}%</h3>
                </div>
              </div>
            </div>
          </div>
          
          {/* Recent Analyses */}
          <div className="glass rounded-xl overflow-hidden mb-8">
            <div className="p-6 border-b border-muted">
              <h2 className="text-xl font-semibold">Recent Analyses</h2>
            </div>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>File Name</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead>Result</TableHead>
                    <TableHead>Confidence</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {recentAnalyses.map((analysis) => (
                    <TableRow key={analysis.id}>
                      <TableCell className="font-medium">{analysis.fileName}</TableCell>
                      <TableCell>{formatDate(analysis.uploadDate)}</TableCell>
                      <TableCell>
                        {analysis.isDeepfake ? (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-destructive/10 text-destructive">
                            Deepfake
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-500/10 text-green-500">
                            Authentic
                          </span>
                        )}
                      </TableCell>
                      <TableCell>{analysis.confidence}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
          
          {/* Deepfake Types */}
          <div className="glass rounded-xl overflow-hidden">
            <div className="p-6 border-b border-muted">
              <h2 className="text-xl font-semibold">Deepfake Types</h2>
            </div>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Type</TableHead>
                    <TableHead>Count</TableHead>
                    <TableHead>Percentage</TableHead>
                    <TableHead>Distribution</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {deepfakeTypes.map((type) => (
                    <TableRow key={type.type}>
                      <TableCell className="font-medium">{type.type}</TableCell>
                      <TableCell>{type.count}</TableCell>
                      <TableCell>{type.percentage}%</TableCell>
                      <TableCell>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div
                            className="bg-primary rounded-full h-2"
                            style={{ width: `${type.percentage}%` }}
                          ></div>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}