import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Upload } from "lucide-react";
import { useLocation } from "wouter";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { VideoAnalysisResult } from "@shared/schema";
import { useQuery } from "@tanstack/react-query";

interface AnalysisItem {
  id: string;
  fileName: string;
  uploadDate: string;
  fileSize?: number;
  duration?: string;
  result: string;
  confidence: number;
  thumbnail?: string;
}

export default function HistoryPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [, navigate] = useLocation();
  const [searchTerm, setSearchTerm] = useState("");
  const [resultFilter, setResultFilter] = useState("all");
  const [timeFilter, setTimeFilter] = useState("all");
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisItem[]>([]);
  
  // Fetch real video analyses from the API
  const { data: videoAnalyses, isLoading, error } = useQuery<VideoAnalysisResult[]>({
    queryKey: ["/api/videos"],
    enabled: !!user
  });

  // Process video analyses data when it loads
  useEffect(() => {
    if (videoAnalyses && videoAnalyses.length > 0) {
      const processedAnalyses = videoAnalyses.map(analysis => {
        // Extract video duration if available (would be stored in analysis data in a real implementation)
        const duration = "2:30"; // This would be extracted from actual video metadata
        
        return {
          id: analysis.id,
          fileName: analysis.fileName,
          uploadDate: analysis.uploadDate,
          fileSize: analysis.fileSize,
          duration: duration,
          result: analysis.analysis.isDeepfake ? "deepfake" : analysis.analysis.confidence > 30 ? "authentic" : "inconclusive",
          confidence: analysis.analysis.confidence,
          // In a real implementation, thumbnails would be generated and stored
          thumbnail: "/thumbnails/default.jpg"
        };
      });
      setAnalysisHistory(processedAnalyses);
    } else {
      setAnalysisHistory([]);
    }
  }, [videoAnalyses]);

  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  // Format time for display
  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString("en-US", { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  // Get badge for analysis result
  const getResultBadge = (result: string) => {
    switch (result) {
      case "deepfake":
        return <Badge variant="destructive">{result}</Badge>;
      case "authentic":
        return <Badge variant="default" className="bg-green-600 hover:bg-green-700">{result}</Badge>;
      case "inconclusive":
        return <Badge variant="secondary">{result}</Badge>;
      default:
        return <Badge>{result}</Badge>;
    }
  };

  // Filter history based on search term and filters
  const filteredHistory = analysisHistory.filter(item => {
    const matchesSearch = 
      item.fileName.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesResult = resultFilter === "all" || item.result === resultFilter;
    
    let matchesTime = true;
    const itemDate = new Date(item.uploadDate);
    const now = new Date();
    
    if (timeFilter === "today") {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      matchesTime = itemDate >= today;
    } else if (timeFilter === "week") {
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      matchesTime = itemDate >= weekAgo;
    } else if (timeFilter === "month") {
      const monthAgo = new Date();
      monthAgo.setMonth(monthAgo.getMonth() - 1);
      matchesTime = itemDate >= monthAgo;
    }
    
    return matchesSearch && matchesResult && matchesTime;
  });

  // Calculate real statistics from the data
  const stats = {
    totalAnalyses: analysisHistory.length,
    deepfakes: analysisHistory.filter(item => item.result === "deepfake").length,
    authentic: analysisHistory.filter(item => item.result === "authentic").length,
    inconclusive: analysisHistory.filter(item => item.result === "inconclusive").length,
    averageConfidence: analysisHistory.length > 0 
      ? Math.round(analysisHistory.reduce((sum, item) => sum + item.confidence, 0) / analysisHistory.length) 
      : 0
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Page Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">Analysis History</h1>
            <p className="text-muted-foreground">View your past video analyses and results</p>
          </div>
          
          <div className="flex items-center gap-3">
            <Button
              onClick={() => navigate("/upload")}
              className="flex items-center gap-2"
            >
              <Upload className="h-4 w-4" />
              Upload New Video
            </Button>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          {isLoading ? (
            Array(5).fill(0).map((_, index) => (
              <Card key={index}>
                <CardContent className="p-4 flex flex-col items-center justify-center min-h-[80px]">
                  <div className="animate-pulse h-8 w-12 bg-muted rounded mb-2"></div>
                  <div className="animate-pulse h-4 w-24 bg-muted rounded"></div>
                </CardContent>
              </Card>
            ))
          ) : (
            <>
              <Card>
                <CardContent className="p-4 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold">{stats.totalAnalyses}</span>
                  <span className="text-sm text-muted-foreground">Total Analyses</span>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold text-red-500">{stats.deepfakes}</span>
                  <span className="text-sm text-muted-foreground">Deepfakes Detected</span>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold text-green-600">{stats.authentic}</span>
                  <span className="text-sm text-muted-foreground">Authentic Videos</span>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold text-gray-400">{stats.inconclusive}</span>
                  <span className="text-sm text-muted-foreground">Inconclusive</span>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold">{stats.averageConfidence}%</span>
                  <span className="text-sm text-muted-foreground">Avg. Confidence</span>
                </CardContent>
              </Card>
            </>
          )}
        </div>

        {/* Filters and Search */}
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="w-full md:w-1/3">
            <Input 
              placeholder="Search by filename..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full"
            />
          </div>
          
          <div className="w-full md:w-1/4">
            <Select value={resultFilter} onValueChange={setResultFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by result" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Results</SelectItem>
                <SelectItem value="deepfake">Deepfakes</SelectItem>
                <SelectItem value="authentic">Authentic</SelectItem>
                <SelectItem value="inconclusive">Inconclusive</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="w-full md:w-1/4">
            <Select value={timeFilter} onValueChange={setTimeFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by time" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Time</SelectItem>
                <SelectItem value="today">Today</SelectItem>
                <SelectItem value="week">Past Week</SelectItem>
                <SelectItem value="month">Past Month</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Grid View */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {isLoading ? (
            // Loading skeleton cards for grid view
            Array(8).fill(0).map((_, index) => (
              <Card key={`loading-grid-${index}`} className="overflow-hidden">
                <div className="h-48 bg-muted/40 relative">
                  <div className="animate-pulse bg-muted h-full w-full"></div>
                  <div className="absolute top-2 right-2">
                    <div className="animate-pulse h-6 w-20 bg-muted/80 rounded-full"></div>
                  </div>
                </div>
              </Card>
            ))
          ) : filteredHistory.length > 0 ? (
            filteredHistory.map((analysis) => (
              <Card key={analysis.id} className="overflow-hidden hover:border-primary/50 transition-colors cursor-pointer">
                <div 
                  className="h-48 bg-muted/40 relative"
                  style={{ 
                    backgroundImage: `url(${analysis.thumbnail || '/thumbnails/default.jpg'})`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center'
                  }}
                >
                  <div className="absolute inset-0 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white opacity-80">
                      <circle cx="12" cy="12" r="10"/>
                      <polygon points="10 8 16 12 10 16 10 8"/>
                    </svg>
                  </div>
                  <div className="absolute top-2 right-2">
                    {getResultBadge(analysis.result)}
                  </div>
                  <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-xs p-2">
                    <div className="flex justify-between items-center">
                      <span>{analysis.duration || 'N/A'}</span>
                      <span>{analysis.fileSize ? `${analysis.fileSize} MB` : 'N/A'}</span>
                    </div>
                  </div>
                </div>
                <CardContent className="p-4">
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex-1 pr-2">
                      <h3 className="font-medium truncate">{analysis.fileName}</h3>
                      <p className="text-xs text-muted-foreground">{formatDate(analysis.uploadDate)}</p>
                    </div>
                    <div className="flex-shrink-0 h-8 w-8 rounded-full bg-muted/70 flex items-center justify-center text-primary">
                      {analysis.result === "deepfake" ? (
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                          <path d="M12 9v4"/>
                          <path d="M12 17h.01"/>
                        </svg>
                      ) : analysis.result === "authentic" ? (
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                          <polyline points="22 4 12 14.01 9 11.01"/>
                        </svg>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="12" cy="12" r="10"/>
                          <path d="M12 8v4"/>
                          <path d="M12 16h.01"/>
                        </svg>
                      )}
                    </div>
                  </div>
                  <div className="mt-2">
                    <div className="text-xs font-medium flex justify-between mb-1">
                      <span>Confidence</span>
                      <span>{Math.round(analysis.confidence * 100)}%</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-1.5">
                      <div 
                        className={`h-1.5 rounded-full ${
                          (analysis.confidence * 100) > 90 ? 'bg-green-500' : 
                          (analysis.confidence * 100) > 70 ? 'bg-yellow-500' : 'bg-red-500'
                        }`} 
                        style={{ width: `${Math.round(analysis.confidence * 100)}%` }}
                      />
                    </div>
                  </div>
                </CardContent>
                <div className="px-4 pb-3 pt-0 flex justify-between border-t">
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="flex-1 mr-1"
                    onClick={() => {
                      navigate(`/analysis/${analysis.id}`);
                    }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 mr-1">
                      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                      <circle cx="12" cy="12" r="3"/>
                    </svg>
                    View
                  </Button>
                </div>
              </Card>
            ))
          ) : (
            <div className="col-span-full text-center py-12 text-muted-foreground">
              {error ? (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-4 opacity-50">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12" y2="16"/>
                  </svg>
                  <p className="text-lg mb-2">Error loading analyses</p>
                  <p className="text-sm">Please try again</p>
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-4 opacity-50">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="9" cy="9" r="2"/>
                    <path d="M21 15l-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
                  </svg>
                  <p className="text-lg mb-2">
                    {searchTerm || resultFilter !== "all" || timeFilter !== "all" ?
                      "No analyses found matching your filters" :
                      "You haven't analyzed any videos yet"}
                  </p>
                  <p className="text-sm mb-4">
                    {searchTerm || resultFilter !== "all" || timeFilter !== "all" ?
                      "Try adjusting your search or filters" :
                      "Upload a video to get started with deepfake detection"}
                  </p>
                  {!(searchTerm || resultFilter !== "all" || timeFilter !== "all") && (
                    <Button
                      onClick={() => navigate("/upload")}
                      className="mt-2"
                    >
                      <Upload className="h-4 w-4 mr-2" />
                      Upload Your First Video
                    </Button>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}