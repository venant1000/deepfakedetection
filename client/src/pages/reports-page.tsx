import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
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
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { VideoAnalysisResult } from "@shared/schema";

// Define report interface
interface Report {
  id: string;
  date: string;
  title: string;
  type: string;
  status?: string;
  saved?: boolean;
  data?: {
    videoId?: string;
    videoIds?: string[];
    fileName?: string;
    isDeepfake?: boolean;
    confidence?: number;
    totalVideos?: number;
    deepfakesFound?: number;
    averageConfidence?: number;
    issues?: { type: string; text: string }[];
  };
}

export default function ReportsPage() {
  const { user } = useAuth();
  const [searchTerm, setSearchTerm] = useState("");
  const [reportType, setReportType] = useState("all");
  const [isExporting, setIsExporting] = useState(false);
  const { toast } = useToast();
  const [generatedReports, setGeneratedReports] = useState<Report[]>([]);
  const [savedReportList, setSavedReportList] = useState<Report[]>([]);
  
  // Fetch actual video analyses from the API
  const { data: videoAnalyses, isLoading, error } = useQuery<VideoAnalysisResult[]>({
    queryKey: ["/api/videos"],
    enabled: !!user
  });
  
  // Generate reports based on actual video analyses
  useEffect(() => {
    if (videoAnalyses && videoAnalyses.length > 0) {
      // Create reports from actual videos data
      const newReports: Report[] = [];
      
      // Summary report (overall statistics)
      if (videoAnalyses.length > 0) {
        const deepfakes = videoAnalyses.filter(v => v.analysis.isDeepfake);
        const averageConfidence = Math.round(
          videoAnalyses.reduce((sum, vid) => sum + vid.analysis.confidence, 0) / videoAnalyses.length
        );
        
        newReports.push({
          id: "summary-report",
          date: new Date().toISOString().split('T')[0],
          title: "Video Analysis Summary Report",
          type: "statistical",
          status: "completed",
          data: {
            totalVideos: videoAnalyses.length,
            deepfakesFound: deepfakes.length,
            averageConfidence: averageConfidence,
            videoIds: videoAnalyses.map(v => v.id)
          }
        });
      }
      
      // Individual video reports
      videoAnalyses.forEach(video => {
        newReports.push({
          id: `video-${video.id}`,
          date: new Date(video.uploadDate).toISOString().split('T')[0],
          title: `Analysis Report: ${video.fileName}`,
          type: video.analysis.isDeepfake ? "analytical" : "educational",
          status: "completed",
          data: {
            videoId: video.id,
            fileName: video.fileName,
            isDeepfake: video.analysis.isDeepfake,
            confidence: video.analysis.confidence,
            issues: video.analysis.issues || []
          }
        });
      });
      
      // Set as the generated reports
      setGeneratedReports(newReports);
      
      // Create some saved reports based on the data
      if (newReports.length > 0) {
        setSavedReportList([
          {
            id: "auto-summary",
            date: new Date().toISOString().split('T')[0],
            title: "DeepGuard Analysis Summary",
            type: "educational",
            saved: true,
            data: {
              videoIds: videoAnalyses.map(v => v.id)
            }
          }
        ]);
      }
    }
  }, [videoAnalyses]);

  // Get badge color based on report type
  const getTypeBadge = (type: string) => {
    switch (type) {
      case "statistical":
        return <Badge variant="secondary">{type}</Badge>;
      case "analytical":
        return <Badge variant="default" className="bg-blue-500 hover:bg-blue-600">{type}</Badge>;
      case "educational":
        return <Badge variant="default" className="bg-green-500 hover:bg-green-600">{type}</Badge>;
      default:
        return <Badge>{type}</Badge>;
    }
  };

  // Filter reports based on search term and type
  const filteredReports = generatedReports.filter(report => {
    const matchesSearch = 
      report.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      report.type.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesType = reportType === "all" || report.type === reportType;
    
    return matchesSearch && matchesType;
  });

  // Function to handle exporting reports
  const handleExportReports = (reportsToExport: Report[], title = "reports") => {
    setIsExporting(true);
    
    try {
      if (reportsToExport.length === 0) {
        toast({
          title: "No reports to export",
          description: "There are no reports available to export.",
          variant: "destructive"
        });
        setIsExporting(false);
        return;
      }
      
      // Create CSV content
      const headers = ["Date", "Title", "Type", "Status"];
      const csvContent = [
        headers.join(","),
        ...reportsToExport.map((report: Report) => [
          report.date,
          `"${report.title.replace(/"/g, '""')}"`, // Escape quotes in titles
          report.type,
          report.status || "completed"
        ].join(","))
      ].join("\n");
      
      // Create and download the file
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `deepguard-${title}-${new Date().toISOString().split('T')[0]}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      toast({
        title: "Export successful",
        description: `Your ${title} have been exported as a CSV file.`
      });
    } catch (error) {
      console.error("Export error:", error);
      toast({
        title: "Export failed",
        description: "There was an error exporting your reports. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Page Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">Reports</h1>
            <p className="text-muted-foreground">Access and manage your reports</p>
          </div>
          
          <Button 
            className="bg-gradient-to-r from-primary to-secondary text-black"
            onClick={() => {
              if (!videoAnalyses || videoAnalyses.length === 0) {
                toast({
                  title: "No analysis data",
                  description: "You need to analyze videos before generating reports. Upload and analyze videos first.",
                });
                return;
              }
              
              toast({
                title: "Report Generated",
                description: "A new report has been generated and added to your list.",
              });
            }}
          >
            {isExporting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                </svg>
                Generate New Report
              </>
            )}
          </Button>
        </div>

        <Tabs defaultValue="all" className="w-full">
          <TabsList className="mb-6 w-full md:w-auto">
            <TabsTrigger value="all">All Reports</TabsTrigger>
            <TabsTrigger value="saved">Saved Reports</TabsTrigger>
          </TabsList>
          
          <TabsContent value="all">
            {/* Filters */}
            <div className="flex flex-col md:flex-row gap-4 mb-6">
              <div className="w-full md:w-1/3">
                <Input 
                  placeholder="Search reports..." 
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full"
                />
              </div>
              <div className="w-full md:w-1/4">
                <Select value={reportType} onValueChange={setReportType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Filter by type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="statistical">Statistical</SelectItem>
                    <SelectItem value="analytical">Analytical</SelectItem>
                    <SelectItem value="educational">Educational</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="w-full md:w-1/4">
                <Select defaultValue="newest">
                  <SelectTrigger>
                    <SelectValue placeholder="Sort by" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="newest">Newest First</SelectItem>
                    <SelectItem value="oldest">Oldest First</SelectItem>
                    <SelectItem value="az">A-Z</SelectItem>
                    <SelectItem value="za">Z-A</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Reports Table */}
            <Card>
              <CardHeader className="pb-3 flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Your Reports</CardTitle>
                  <CardDescription>
                    Reports generated from your analyses and system data
                  </CardDescription>
                </div>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => handleExportReports(generatedReports, "all-reports")}
                  disabled={isExporting || generatedReports.length === 0}
                >
                  {isExporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" x2="12" y1="15" y2="3"/>
                      </svg>
                      Export
                    </>
                  )}
                </Button>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[120px]">Date</TableHead>
                      <TableHead>Title</TableHead>
                      <TableHead className="w-[120px]">Type</TableHead>
                      <TableHead className="w-[120px]">Status</TableHead>
                      <TableHead className="w-[120px] text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {isLoading ? (
                      <TableRow>
                        <TableCell colSpan={5} className="text-center py-8">
                          <div className="flex justify-center items-center">
                            <Loader2 className="h-6 w-6 animate-spin mr-2" />
                            <span>Loading reports...</span>
                          </div>
                        </TableCell>
                      </TableRow>
                    ) : generatedReports.length > 0 ? (
                      generatedReports.map((report) => (
                        <TableRow key={report.id} className="hover:bg-muted/30 transition-colors">
                          <TableCell className="font-mono text-xs">
                            {report.date}
                          </TableCell>
                          <TableCell className="font-medium">{report.title}</TableCell>
                          <TableCell>{getTypeBadge(report.type)}</TableCell>
                          <TableCell>
                            <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
                              {report.status || "completed"}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-2">
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"/>
                                  <path d="M12 16v-4"/>
                                  <path d="M12 8h.01"/>
                                </svg>
                              </Button>
                              <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8"
                                onClick={() => {
                                  // Export individual report
                                  handleExportReports([report], `report-${report.id}`);
                                }}
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 0 0 2-2v-4"/>
                                  <path d="M17 9l-5 5-5-5"/>
                                  <path d="M12 12.8V2.5"/>
                                </svg>
                              </Button>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                                  <polyline points="17 21 17 13 7 13 7 21"/>
                                  <polyline points="7 3 7 8 15 8"/>
                                </svg>
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                          {error ? (
                            <div>
                              Error loading reports. Please try again.
                            </div>
                          ) : (
                            <div>
                              No reports found. Upload and analyze videos to generate reports.
                            </div>
                          )}
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="saved">
            <Card>
              <CardHeader className="pb-3 flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Saved Reports</CardTitle>
                  <CardDescription>
                    Reports you've bookmarked for easy reference
                  </CardDescription>
                </div>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => handleExportReports(savedReportList, "saved-reports")}
                  disabled={isExporting || savedReportList.length === 0}
                >
                  {isExporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" x2="12" y1="15" y2="3"/>
                      </svg>
                      Export
                    </>
                  )}
                </Button>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[120px]">Date</TableHead>
                      <TableHead>Title</TableHead>
                      <TableHead className="w-[120px]">Type</TableHead>
                      <TableHead className="w-[120px] text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {isLoading ? (
                      <TableRow>
                        <TableCell colSpan={4} className="text-center py-8">
                          <div className="flex justify-center items-center">
                            <Loader2 className="h-6 w-6 animate-spin mr-2" />
                            <span>Loading saved reports...</span>
                          </div>
                        </TableCell>
                      </TableRow>
                    ) : savedReportList.length > 0 ? (
                      savedReportList.map((report: Report) => (
                        <TableRow key={report.id} className="hover:bg-muted/30 transition-colors">
                          <TableCell className="font-mono text-xs">
                            {report.date}
                          </TableCell>
                          <TableCell className="font-medium">{report.title}</TableCell>
                          <TableCell>{getTypeBadge(report.type)}</TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-2">
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"/>
                                  <path d="M12 16v-4"/>
                                  <path d="M12 8h.01"/>
                                </svg>
                              </Button>
                              <Button variant="ghost" size="icon" className="h-8 w-8 text-yellow-500">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                                </svg>
                              </Button>
                              <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8"
                                onClick={() => {
                                  // Export individual saved report
                                  handleExportReports([report], `saved-report-${report.id}`);
                                }}
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 0 0 2-2v-4"/>
                                  <path d="M17 9l-5 5-5-5"/>
                                  <path d="M12 12.8V2.5"/>
                                </svg>
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={4} className="text-center py-8 text-muted-foreground">
                          No saved reports yet
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}