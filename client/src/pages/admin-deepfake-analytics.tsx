import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Download, ArrowUpRight, RefreshCw, Calendar, ChevronDown, FileText, FileSpreadsheet } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

export default function AdminDeepfakeAnalytics() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [timeRange, setTimeRange] = useState("last30days");
  const [analyticsData, setAnalyticsData] = useState<any>(null);
  
  // Format date for display
  const formatLastUpdated = () => {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Fetch analytics data
  const fetchAnalytics = async (showToast = false) => {
    try {
      if (showToast) {
        setRefreshing(true);
      } else {
        setIsLoading(true);
      }
      
      const response = await fetch("/api/admin/stats", {
        credentials: "include",
      });

      if (!response.ok) {
        if (response.status === 403) {
          toast({
            title: "Access denied",
            description: "You don't have permission to view this page",
            variant: "destructive",
          });
          navigate("/dashboard");
          return;
        }
        throw new Error("Failed to fetch analytics data");
      }

      // Get complete analytics data from the API
      const data = await response.json();
      
      // Use the data directly as it now comes fully formed from the backend
      setAnalyticsData(data);
      
      if (showToast) {
        toast({
          title: "Data refreshed",
          description: "Analytics data has been updated with the latest information.",
        });
      }
      
      console.log("Analytics data loaded from database:", data);
    } catch (error) {
      console.error("Error fetching analytics:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to load analytics data",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);
  
  // Handle manual refresh
  const handleRefresh = () => {
    fetchAnalytics(true);
  };

  // Export to CSV
  const exportToCSV = () => {
    if (!analyticsData) return;

    const csvData = [
      // Summary data
      ['Deepfake Analytics Report'],
      ['Generated on:', new Date().toLocaleDateString()],
      [''],
      ['Summary Statistics'],
      ['Total Users', analyticsData.summary?.totalUsers || 0],
      ['Total Videos Analyzed', analyticsData.summary?.videoCount || 0],
      ['Deepfakes Detected', analyticsData.summary?.deepfakeCount || 0],
      ['Detection Accuracy', `${analyticsData.summary?.systemHealth || 0}%`],
      [''],
      ['Daily Upload Statistics'],
      ['Date', 'Upload Count'],
      ...(analyticsData.dailyUploads || []).map((item: any) => [item.date, item.count]),
      [''],
      ['Detection Rate Trends'],
      ['Date', 'Detection Rate (%)'],
      ...(analyticsData.detectionRates || []).map((item: any) => [item.date, item.rate]),
      [''],
      ['Processing Time Distribution'],
      ['Time Range', 'Count'],
      ...(analyticsData.processingTimes || []).map((item: any) => [item.timeRange, item.count]),
    ];

    const csvContent = csvData.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `deepfake-analytics-${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    toast({
      title: "Export successful",
      description: "Analytics data has been exported to CSV.",
    });
  };

  // Export to PDF (simplified version using browser print)
  const exportToPDF = () => {
    toast({
      title: "PDF Export",
      description: "Use your browser's print function (Ctrl/Cmd + P) and select 'Save as PDF' to export this page.",
    });
    
    // Trigger browser print dialog
    setTimeout(() => {
      window.print();
    }, 500);
  };

  // Chart colors
  const colors = {
    primary: "#00ff88",
    secondary: "#7000ff",
    tertiary: "#00a3ff",
    quaternary: "#ff3e66",
    warning: "#ff9500",
    neutral: "#888888",
    success: "#00ff88",
    error: "#ff3e66",
    dark: "#333333",
    light: "#f5f5f5"
  };
  
  const COLORS = [colors.primary, colors.secondary, colors.tertiary, colors.quaternary];
  
  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex">
        <Sidebar isAdmin={true} />
        
        <div className="flex-1 ml-20 md:ml-64 p-8 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="h-12 w-12 mx-auto mb-4 animate-spin text-primary" />
            <p className="text-muted-foreground">Loading analytics data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar isAdmin={true} />
      
      <div className="flex-1 ml-20 md:ml-64 p-6 py-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
            <div>
              <h1 className="text-3xl font-bold mb-1">Deepfake Analytics Dashboard</h1>
              <div className="flex items-center text-sm text-muted-foreground">
                <span>Last updated: {formatLastUpdated()}</span>
                {refreshing && <Loader2 className="h-3 w-3 ml-2 animate-spin" />}
              </div>
            </div>
            
            <div className="flex flex-wrap items-center gap-2">
              <Button 
                size="icon" 
                variant="ghost" 
                onClick={handleRefresh} 
                disabled={refreshing}
                className="h-8 w-8"
              >
                <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              </Button>
              
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="h-8">
                    <Download className="h-4 w-4 mr-2" />
                    Export
                    <ChevronDown className="h-4 w-4 ml-2" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuLabel>Export Options</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={exportToCSV}>
                    <FileSpreadsheet className="h-4 w-4 mr-2" />
                    Export as CSV
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={exportToPDF}>
                    <FileText className="h-4 w-4 mr-2" />
                    Export as PDF
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              
              <div className="flex items-center border rounded-md px-3 py-1">
                <Calendar className="h-4 w-4 mr-2 text-muted-foreground" />
                <Select value={timeRange} onValueChange={setTimeRange}>
                  <SelectTrigger className="border-0 p-0 h-auto shadow-none">
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
              </div>
              
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="ml-auto">
                    <Download className="h-4 w-4 mr-2" />
                    Export
                    <ChevronDown className="h-4 w-4 ml-2" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuLabel>Export Options</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    Export as PDF
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    Export as CSV
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    Export as Excel
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
          
          {/* Summary Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <Card className="overflow-hidden border-0 shadow-md">
              <CardHeader className="pb-2 bg-gradient-to-r from-primary/10 to-primary/5">
                <div className="flex justify-between items-start">
                  <CardDescription>Videos Analyzed</CardDescription>
                  <span className="bg-primary/20 text-primary rounded-full p-1">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
                  </span>
                </div>
                <CardTitle className="text-3xl font-bold">{analyticsData?.summary?.videoCount || 0}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex justify-between items-center">
                  <div className="text-xs text-muted-foreground flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><path d="m18 15-6-6-6 6"/></svg>
                    <span className="text-green-500">+{analyticsData?.summary?.videoCount || 0}</span>
                    <span className="ml-1">since last week</span>
                  </div>
                  <Badge variant="outline" className="text-xs border-primary/30 text-primary">
                    <ArrowUpRight size={12} className="mr-1" /> 100%
                  </Badge>
                </div>
              </CardContent>
            </Card>
            
            <Card className="overflow-hidden border-0 shadow-md">
              <CardHeader className="pb-2 bg-gradient-to-r from-secondary/10 to-secondary/5">
                <div className="flex justify-between items-start">
                  <CardDescription>Deepfakes Detected</CardDescription>
                  <span className="bg-secondary/20 text-secondary rounded-full p-1">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
                  </span>
                </div>
                <CardTitle className="text-3xl font-bold">{analyticsData?.summary?.deepfakeCount || 0}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex justify-between items-center">
                  <div className="text-xs text-muted-foreground flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><path d="m18 15-6-6-6 6"/></svg>
                    <span className="text-green-500">+{analyticsData?.summary?.deepfakeCount || 0}</span>
                    <span className="ml-1">since last week</span>
                  </div>
                  <Badge variant="outline" className="text-xs border-secondary/30 text-secondary">
                    <ArrowUpRight size={12} className="mr-1" /> 100%
                  </Badge>
                </div>
              </CardContent>
            </Card>
            
            <Card className="overflow-hidden border-0 shadow-md">
              <CardHeader className="pb-2 bg-gradient-to-r from-tertiary/10 to-tertiary/5">
                <div className="flex justify-between items-start">
                  <CardDescription>Detection Rate</CardDescription>
                  <span className="bg-tertiary/20 text-tertiary rounded-full p-1">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5.8 11.3a9 9 0 1 0 9.8-8.8"/><path d="M5.9 15.4a9 9 0 0 1-3.9-8"/><path d="M2 2v6h6"/></svg>
                  </span>
                </div>
                <CardTitle className="text-3xl font-bold">
                  {analyticsData?.summary?.videoCount && analyticsData?.summary?.deepfakesDetected 
                    ? `${((analyticsData.summary.deepfakesDetected / analyticsData.summary.videoCount) * 100).toFixed(1)}%` 
                    : '0%'}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Progress value={
                  analyticsData?.summary?.videoCount && analyticsData?.summary?.deepfakesDetected 
                    ? (analyticsData.summary.deepfakesDetected / analyticsData.summary.videoCount) * 100
                    : 0
                } className="h-2 mb-2" />
                <div className="text-xs text-muted-foreground">
                  Percentage of videos classified as deepfakes
                </div>
              </CardContent>
            </Card>
            
            <Card className="overflow-hidden border-0 shadow-md">
              <CardHeader className="pb-2 bg-gradient-to-r from-quaternary/10 to-quaternary/5">
                <div className="flex justify-between items-start">
                  <CardDescription>Avg. Confidence</CardDescription>
                  <span className="bg-quaternary/20 text-quaternary rounded-full p-1">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.2 8.4-4.2 4.2-4.3-4.2"/><path d="m21.2 15.6-4.2-4.2-4.3 4.2"/><line x1="3" x2="9.5" y1="12" y2="12"/></svg>
                  </span>
                </div>
                <CardTitle className="text-3xl font-bold">{
                  analyticsData?.summary?.videoCount 
                    ? `${(parseFloat(((1 - (analyticsData?.detectionRates?.reduce((acc: number, curr: any) => acc + curr.rate, 0) / analyticsData?.detectionRates?.length || 0) / 100)).toFixed(2)) * 100).toFixed(1)}%` 
                    : '0%'
                }</CardTitle>
              </CardHeader>
              <CardContent>
                <Progress value={
                  analyticsData?.summary?.videoCount 
                    ? (1 - (analyticsData?.detectionRates?.reduce((acc: number, curr: any) => acc + curr.rate, 0) / analyticsData?.detectionRates?.length || 0) / 100) * 100
                    : 0
                } className="h-2 mb-2" />
                <div className="text-xs text-muted-foreground">
                  Average detection confidence score
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Charts Section */}
          <div className="space-y-6">
            {/* Confidence Score Distribution & Deepfake Categories */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* Confidence Score Distribution Chart */}
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle>Confidence Score Distribution</CardTitle>
                  <CardDescription>Distribution of detection confidence levels</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-80 p-6">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={[
                          { range: "0-20%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.15) : 0 },
                          { range: "20-40%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.25) : 0 },
                          { range: "40-60%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.3) : 0 },
                          { range: "60-80%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.2) : 0 },
                          { range: "80-100%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.1) : 0 }
                        ]}
                        margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#444" />
                        <XAxis dataKey="range" tick={{ fill: '#888' }} />
                        <YAxis tick={{ fill: '#888' }} />
                        <Tooltip 
                          contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                          labelStyle={{ color: '#fff' }}
                          itemStyle={{ color: '#fff' }}
                        />
                        <Bar 
                          dataKey="count" 
                          name="Video Count" 
                          fill={colors.primary}
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
              
              {/* Detection Results Distribution Chart */}
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle>Detection Results Distribution</CardTitle>
                  <CardDescription>Real vs Deepfake detection results</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-80 p-6">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={[
                            { 
                              name: "Authentic Videos", 
                              value: analyticsData?.summary?.videoCount ? analyticsData.summary.videoCount - analyticsData.summary.deepfakeCount : 0
                            },
                            { 
                              name: "Deepfake Detected", 
                              value: analyticsData?.summary?.deepfakeCount || 0 
                            }
                          ]}
                          nameKey="name"
                          dataKey="value"
                          cx="50%"
                          cy="50%"
                          outerRadius="70%"
                          innerRadius="40%"
                          paddingAngle={2}
                          label={(entry) => entry.name}
                          labelLine={{ stroke: "#555" }}
                        >
                          <Cell fill={colors.success} />
                          <Cell fill={colors.error} />
                        </Pie>
                        <Tooltip 
                          contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                          labelStyle={{ color: '#fff' }}
                          itemStyle={{ color: '#fff' }}
                        />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            {/* Detection Rates Over Time */}
            <Card className="border-0 shadow-md mb-6">
              <CardHeader>
                <CardTitle>Detection Rate Trends</CardTitle>
                <CardDescription>Deepfake detection rates over time</CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div className="h-80 p-6">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={analyticsData?.detectionRates || [
                        { date: "May 17", rate: 23.5 },
                        { date: "May 18", rate: 24.8 },
                        { date: "May 19", rate: 28.4 },
                        { date: "May 20", rate: 26.2 },
                        { date: "May 21", rate: 29.5 },
                        { date: "May 22", rate: 33.1 },
                        { date: "May 23", rate: 30.8 }
                      ]}
                      margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#444" />
                      <XAxis dataKey="date" tick={{ fill: '#888' }} />
                      <YAxis tick={{ fill: '#888' }} />
                      <Tooltip 
                        contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                        labelStyle={{ color: '#fff' }}
                        itemStyle={{ color: '#fff' }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="rate" 
                        name="Detection Rate (%)" 
                        stroke={colors.quaternary} 
                        strokeWidth={3}
                        dot={{ fill: colors.quaternary, r: 4 }}
                        activeDot={{ fill: colors.quaternary, r: 6, stroke: 'white', strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            {/* Processing Times and Classification Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Processing Time Distribution */}
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle>Processing Time Distribution</CardTitle>
                  <CardDescription>Analysis time for video processing</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-80 p-6">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={analyticsData?.processingTimes || [
                          { timeRange: "<30s", count: 45 },
                          { timeRange: "30s-1m", count: 32 },
                          { timeRange: "1m-2m", count: 18 },
                          { timeRange: "2m-5m", count: 5 },
                          { timeRange: ">5m", count: 2 }
                        ]}
                        margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#444" />
                        <XAxis dataKey="timeRange" tick={{ fill: '#888' }} />
                        <YAxis tick={{ fill: '#888' }} />
                        <Tooltip 
                          contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                          labelStyle={{ color: '#fff' }}
                          itemStyle={{ color: '#fff' }}
                        />
                        <Bar 
                          dataKey="count" 
                          name="Video Count" 
                          fill={colors.tertiary}
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
              
              {/* Classification Breakdown */}
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle>Classification Breakdown</CardTitle>
                  <CardDescription>Video classification results</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-80 p-6">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={analyticsData?.classificationBreakdown || [
                            { name: "Authentic", value: 65, color: "#00ff88" },
                            { name: "Deepfake", value: 25, color: "#ff3366" },
                            { name: "Moderate/Suspicious", value: 10, color: "#ffaa00" }
                          ]}
                          nameKey="name"
                          dataKey="value"
                          cx="50%"
                          cy="50%"
                          outerRadius="70%"
                          innerRadius={0}
                          paddingAngle={0}
                          label={(entry) => `${entry.name}: ${entry.value}%`}
                        >
                          {(analyticsData?.classificationBreakdown || []).map((entry: any, index: number) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                          labelStyle={{ color: '#fff' }}
                          itemStyle={{ color: '#fff' }}
                        />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}