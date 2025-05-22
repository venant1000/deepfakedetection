import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
        return <Badge variant="default" className="bg-green-500 hover:bg-green-600">{result}</Badge>;
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

  // Handle export functionality
  const handleExportHistory = () => {
    if (analysisHistory.length === 0) {
      toast({
        title: "No data to export",
        description: "You don't have any analyses to export yet.",
        variant: "destructive"
      });
      return;
    }

    // Create CSV content
    const headers = ["File Name", "Date", "Size (MB)", "Result", "Confidence"];
    const csvContent = [
      headers.join(","),
      ...analysisHistory.map(item => [
        item.fileName,
        formatDate(item.uploadDate),
        item.fileSize || 0,
        item.result,
        item.confidence
      ].join(","))
    ].join("\n");

    // Create and download the file
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `deepfake-analysis-history-${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    toast({
      title: "Export successful",
      description: "Your analysis history has been exported as a CSV file."
    });
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
          
          <Button 
            className="bg-gradient-to-r from-primary to-secondary text-black"
            onClick={handleExportHistory}
            disabled={isLoading || analysisHistory.length === 0}
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading...
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7 10 12 15 17 10"/>
                  <line x1="12" x2="12" y1="15" y2="3"/>
                </svg>
                Export History
              </>
            )}
          </Button>
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
                  <span className="text-3xl font-bold text-green-500">{stats.authentic}</span>
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

        <Tabs defaultValue="table" className="w-full">
          <TabsList className="mb-6 w-full md:w-auto">
            <TabsTrigger value="table">Table View</TabsTrigger>
            <TabsTrigger value="grid">Grid View</TabsTrigger>
          </TabsList>
          
          {/* Table View */}
          <TabsContent value="table">
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[200px]">File Name</TableHead>
                      <TableHead className="w-[130px]">Date</TableHead>
                      <TableHead className="w-[90px]">Size</TableHead>
                      <TableHead className="w-[90px]">Duration</TableHead>
                      <TableHead className="w-[120px]">Result</TableHead>
                      <TableHead className="w-[120px]">Confidence</TableHead>
                      <TableHead className="w-[120px] text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {isLoading ? (
                      // Loading state with skeleton rows
                      Array(5).fill(0).map((_, index) => (
                        <TableRow key={`loading-${index}`}>
                          <TableCell>
                            <div className="animate-pulse h-4 w-32 bg-muted rounded"></div>
                          </TableCell>
                          <TableCell>
                            <div className="animate-pulse h-4 w-24 bg-muted rounded mb-1"></div>
                            <div className="animate-pulse h-3 w-16 bg-muted rounded"></div>
                          </TableCell>
                          <TableCell>
                            <div className="animate-pulse h-4 w-12 bg-muted rounded"></div>
                          </TableCell>
                          <TableCell>
                            <div className="animate-pulse h-4 w-12 bg-muted rounded"></div>
                          </TableCell>
                          <TableCell>
                            <div className="animate-pulse h-5 w-20 bg-muted rounded"></div>
                          </TableCell>
                          <TableCell>
                            <div className="animate-pulse h-3 w-full bg-muted rounded mb-1"></div>
                          </TableCell>
                          <TableCell>
                            <div className="flex justify-end gap-1">
                              <div className="animate-pulse h-6 w-6 bg-muted rounded-full"></div>
                              <div className="animate-pulse h-6 w-6 bg-muted rounded-full"></div>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : filteredHistory.length > 0 ? (
                      filteredHistory.map((analysis) => (
                        <TableRow key={analysis.id} className="hover:bg-muted/30 transition-colors">
                          <TableCell className="font-medium">
                            {analysis.fileName}
                          </TableCell>
                          <TableCell>
                            <div className="text-sm">
                              <div>{formatDate(analysis.uploadDate)}</div>
                              <div className="text-muted-foreground">{formatTime(analysis.uploadDate)}</div>
                            </div>
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {analysis.fileSize ? `${analysis.fileSize} MB` : 'N/A'}
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {analysis.duration || 'N/A'}
                          </TableCell>
                          <TableCell>
                            {getResultBadge(analysis.result)}
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <div className="w-full bg-muted rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full ${
                                    (analysis.confidence * 100) > 90 ? 'bg-green-500' : 
                                    (analysis.confidence * 100) > 70 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`} 
                                  style={{ width: `${Math.round(analysis.confidence * 100)}%` }}
                                />
                              </div>
                              <span className="text-xs font-medium">{Math.round(analysis.confidence * 100)}%</span>
                            </div>
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-2">
                              <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8"
                                onClick={() => {
                                  window.location.href = `/analysis/${analysis.id}`;
                                }}
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                                  <circle cx="12" cy="12" r="3"/>
                                </svg>
                              </Button>
                              <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8"
                                onClick={() => {
                                  // Create a blob URL for downloading the raw analysis data
                                  const data = videoAnalyses?.find(v => v.id === analysis.id);
                                  if (data) {
                                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
                                    const url = URL.createObjectURL(blob);
                                    const link = document.createElement("a");
                                    link.setAttribute("href", url);
                                    link.setAttribute("download", `analysis-${data.id}.json`);
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                    
                                    toast({
                                      title: "Download successful",
                                      description: `Analysis data for ${data.fileName} has been downloaded.`
                                    });
                                  }
                                }}
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                  <polyline points="7 10 12 15 17 10"/>
                                  <line x1="12" x2="12" y1="15" y2="3"/>
                                </svg>
                              </Button>
                              <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8" 
                                onClick={() => {
                                  toast({
                                    title: "Report Feature",
                                    description: "The report feature will be available in a future update."
                                  });
                                }}
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3Z"/>
                                </svg>
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                          {error ? (
                            <>
                              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-2">
                                <circle cx="12" cy="12" r="10"/>
                                <line x1="12" y1="8" x2="12" y2="12"/>
                                <line x1="12" y1="16" x2="12" y2="16"/>
                              </svg>
                              Error loading analyses. Please try again.
                            </>
                          ) : (
                            searchTerm || resultFilter !== "all" || timeFilter !== "all" ?
                              "No analyses found matching your filters" :
                              "You haven't analyzed any videos yet. Upload a video to get started."
                          )}
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Grid View */}
          <TabsContent value="grid">
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
                        className="text-xs"
                        onClick={() => {
                          window.location.href = `/analysis/${analysis.id}`;
                        }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                          <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                          <circle cx="12" cy="12" r="3"/>
                        </svg>
                        View
                      </Button>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="text-xs"
                        onClick={() => {
                          // Download individual analysis data
                          const data = videoAnalyses?.find(v => v.id === analysis.id);
                          if (data) {
                            const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
                            const url = URL.createObjectURL(blob);
                            const link = document.createElement("a");
                            link.setAttribute("href", url);
                            link.setAttribute("download", `analysis-${data.id}.json`);
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            
                            toast({
                              title: "Download successful",
                              description: `Analysis data for ${data.fileName} has been downloaded.`
                            });
                          }
                        }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                          <polyline points="7 10 12 15 17 10"/>
                          <line x1="12" x2="12" y1="15" y2="3"/>
                        </svg>
                        Download
                      </Button>
                    </div>
                  </Card>
                ))
              ) : (
                <div className="col-span-full flex flex-col items-center justify-center py-12 text-center">
                  {error ? (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-red-500 mb-4">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12" y2="16"/>
                      </svg>
                      <h3 className="text-xl font-medium mb-2">Error loading analyses</h3>
                      <p className="text-muted-foreground text-center max-w-md">
                        There was a problem retrieving your analysis history. Please try again later.
                      </p>
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground mb-4">
                        <circle cx="11" cy="11" r="8"/>
                        <path d="m21 21-4.3-4.3"/>
                      </svg>
                      <h3 className="text-xl font-medium mb-2">No results found</h3>
                      <p className="text-muted-foreground text-center max-w-md">
                        {searchTerm || resultFilter !== "all" || timeFilter !== "all" ?
                          "We couldn't find any analysis history that matches your filters. Try changing your search criteria." :
                          "You haven't analyzed any videos yet. Use the upload page to analyze your first video."}
                      </p>
                    </>
                  )}
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}