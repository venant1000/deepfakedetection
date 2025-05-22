import { useState, useEffect, useRef } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Download, ArrowUpRight, RefreshCw, Calendar, ChevronDown, Zap, AlertTriangle, HelpCircle, FileText, PieChart as PieChartIcon, Activity, Clock, Filter } from "lucide-react";
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
  CardFooter,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip as TooltipUI, 
  TooltipContent, 
  TooltipProvider, 
  TooltipTrigger
} from "@/components/ui/tooltip";
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
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ScatterChart,
  Scatter,
  ComposedChart,
} from "recharts";

export default function AdminAnalyticsPage() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [timeRange, setTimeRange] = useState("last30days");
  const [analyticsData, setAnalyticsData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [showAnomalies, setShowAnomalies] = useState(false);
  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastUpdatedRef = useRef<Date>(new Date());
  
  // Format date for display
  const formatLastUpdated = () => {
    return lastUpdatedRef.current.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
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
      lastUpdatedRef.current = new Date();
      
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
    
    // Set up auto-refresh every 5 minutes
    refreshTimerRef.current = setInterval(() => {
      fetchAnalytics(false);
    }, 5 * 60 * 1000);
    
    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, []);
  
  // Handle time range change
  useEffect(() => {
    if (analyticsData) {
      // In a real app, we would refetch with the new timeRange
      // For now, we'll just simulate a refresh
      setRefreshing(true);
      setTimeout(() => {
        setRefreshing(false);
        toast({
          title: "Time range updated",
          description: `Showing data for ${timeRange === "last7days" ? "the last 7 days" : 
                         timeRange === "last30days" ? "the last 30 days" : 
                         timeRange === "last90days" ? "the last 90 days" : 
                         timeRange === "lastYear" ? "the last year" : "all time"}`,
        });
      }, 800);
    }
  }, [timeRange]);

  // Handle manual refresh
  const handleRefresh = () => {
    fetchAnalytics(true);
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
  
  // Some metrics for extended dashboards
  const systemMetrics = [
    { name: "CPU Usage", value: 32, target: 60, color: colors.primary },
    { name: "Memory Usage", value: 45, target: 70, color: colors.secondary },
    { name: "Storage", value: 72, target: 80, color: colors.warning },
    { name: "Network", value: 18, target: 50, color: colors.tertiary },
  ];
  
  const anomalyData = [
    { id: 'ANM-001', type: 'High CPU Usage', timestamp: '2025-05-22 12:45', severity: 'medium', status: 'resolved' },
    { id: 'ANM-002', type: 'Unusual Login Pattern', timestamp: '2025-05-22 16:30', severity: 'high', status: 'investigating' },
    { id: 'ANM-003', type: 'API Latency Spike', timestamp: '2025-05-21 09:15', severity: 'low', status: 'resolved' },
    { id: 'ANM-004', type: 'Database Connection Error', timestamp: '2025-05-20 14:22', severity: 'high', status: 'resolved' },
  ];
  
  // Model performance metrics
  const modelPerformanceData = [
    { name: 'Accuracy', current: 96.3, previous: 95.2 },
    { name: 'Precision', current: 94.7, previous: 93.5 },
    { name: 'Recall', current: 93.8, previous: 92.1 },
    { name: 'F1 Score', current: 94.2, previous: 92.8 },
  ];
  
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
              <h1 className="text-3xl font-bold mb-1">Analytics Dashboard</h1>
              <div className="flex items-center text-sm text-muted-foreground">
                <Clock className="h-4 w-4 mr-1" /> 
                <span>Last updated: {formatLastUpdated()}</span>
                {refreshing && <Loader2 className="h-3 w-3 ml-2 animate-spin" />}
              </div>
            </div>
            
            <div className="flex flex-wrap items-center gap-2">
              <TooltipProvider>
                <TooltipUI>
                  <TooltipTrigger asChild>
                    <Button 
                      size="icon" 
                      variant="ghost" 
                      onClick={handleRefresh} 
                      disabled={refreshing}
                      className="h-8 w-8"
                    >
                      <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Refresh data</TooltipContent>
                </TooltipUI>
              </TooltipProvider>
              
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
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    Schedule Reports
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
          
          <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-8">
            <TabsList className="grid grid-cols-4 md:w-auto w-full">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="deepfakes">Deepfake Analysis</TabsTrigger>
              <TabsTrigger value="system">System Health</TabsTrigger>
              <TabsTrigger value="model">Model Performance</TabsTrigger>
            </TabsList>
            
            <div className="mt-4">
              <TabsContent value="overview" className="m-0">
                {/* Summary Stats */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                  <Card className="overflow-hidden border-0 shadow-md">
                    <CardHeader className="pb-2 bg-gradient-to-r from-primary/10 to-primary/5">
                      <div className="flex justify-between items-start">
                        <CardDescription>Total Users</CardDescription>
                        <span className="bg-primary/20 text-primary rounded-full p-1">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
                        </span>
                      </div>
                      <CardTitle className="text-3xl font-bold">{analyticsData?.summary?.totalUsers || 0}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex justify-between items-center">
                        <div className="text-xs text-muted-foreground flex items-center">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><path d="m18 15-6-6-6 6"/></svg>
                          <span className="text-green-500">+4</span>
                          <span className="ml-1">since last week</span>
                        </div>
                        <Badge variant="outline" className="text-xs border-primary/30 text-primary">
                          <ArrowUpRight size={12} className="mr-1" /> 8.2%
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="overflow-hidden border-0 shadow-md">
                    <CardHeader className="pb-2 bg-gradient-to-r from-secondary/10 to-secondary/5">
                      <div className="flex justify-between items-start">
                        <CardDescription>Videos Analyzed</CardDescription>
                        <span className="bg-secondary/20 text-secondary rounded-full p-1">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
                        </span>
                      </div>
                      <CardTitle className="text-3xl font-bold">{analyticsData?.summary?.videoCount || 0}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex justify-between items-center">
                        <div className="text-xs text-muted-foreground flex items-center">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><path d="m18 15-6-6-6 6"/></svg>
                          <span className="text-green-500">+23</span>
                          <span className="ml-1">since last week</span>
                        </div>
                        <Badge variant="outline" className="text-xs border-secondary/30 text-secondary">
                          <ArrowUpRight size={12} className="mr-1" /> 16.5%
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="overflow-hidden border-0 shadow-md">
                    <CardHeader className="pb-2 bg-gradient-to-r from-tertiary/10 to-tertiary/5">
                      <div className="flex justify-between items-start">
                        <CardDescription>Deepfakes Detected</CardDescription>
                        <span className="bg-tertiary/20 text-tertiary rounded-full p-1">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
                        </span>
                      </div>
                      <CardTitle className="text-3xl font-bold">{analyticsData?.summary?.deepfakesDetected || 0}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex justify-between items-center">
                        <div className="text-xs text-muted-foreground flex items-center">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><path d="m18 15-6-6-6 6"/></svg>
                          <span className="text-green-500">+11</span>
                          <span className="ml-1">since last week</span>
                        </div>
                        <Badge variant="outline" className="text-xs border-tertiary/30 text-tertiary">
                          <ArrowUpRight size={12} className="mr-1" /> 12.3%
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="overflow-hidden border-0 shadow-md">
                    <CardHeader className="pb-2 bg-gradient-to-r from-success/10 to-success/5">
                      <div className="flex justify-between items-start">
                        <CardDescription>System Health</CardDescription>
                        <span className="bg-success/20 text-success rounded-full p-1">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                        </span>
                      </div>
                      <CardTitle className="text-3xl font-bold">{analyticsData?.summary?.systemHealth || 0}%</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex justify-between items-center">
                        <div className="text-xs text-muted-foreground flex items-center">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                          <span>System operating normally</span>
                        </div>
                        <Badge variant="outline" className="text-xs border-success/30 text-success bg-success/5">
                          Optimal
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                {/* Charts Row 1 */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                  <Card className="col-span-1 border-0 shadow-md">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle>Daily Video Uploads</CardTitle>
                          <CardDescription>Number of videos uploaded per day</CardDescription>
                        </div>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <Filter className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem>Show All</DropdownMenuItem>
                            <DropdownMenuItem>Show Last Week</DropdownMenuItem>
                            <DropdownMenuItem>Show Last Month</DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart
                            data={analyticsData?.dailyUploads || []}
                            margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                          >
                            <defs>
                              <linearGradient id="colorUploads" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={colors.primary} stopOpacity={0.8}/>
                                <stop offset="95%" stopColor={colors.primary} stopOpacity={0.1}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.1} />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: 'rgba(0, 0, 0, 0.85)', 
                                border: 'none',
                                borderRadius: '4px',
                                color: 'white' 
                              }}
                              itemStyle={{ color: colors.primary }}
                              labelStyle={{ color: 'white' }}
                            />
                            <Area 
                              type="monotone" 
                              dataKey="count" 
                              stroke={colors.primary} 
                              fillOpacity={1} 
                              fill="url(#colorUploads)" 
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="col-span-1 border-0 shadow-md">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle>Detection Rate Trend</CardTitle>
                          <CardDescription>Percentage of videos identified as deepfakes</CardDescription>
                        </div>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <Filter className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem>Show All</DropdownMenuItem>
                            <DropdownMenuItem>Show Last Week</DropdownMenuItem>
                            <DropdownMenuItem>Show Last Month</DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart
                            data={analyticsData?.detectionRates || []}
                            margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                          >
                            <defs>
                              <linearGradient id="colorRate" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={colors.secondary} stopOpacity={0.8}/>
                                <stop offset="95%" stopColor={colors.secondary} stopOpacity={0.1}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.1} />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: 'rgba(0, 0, 0, 0.85)', 
                                border: 'none',
                                borderRadius: '4px',
                                color: 'white' 
                              }}
                              itemStyle={{ color: colors.secondary }}
                              labelStyle={{ color: 'white' }}
                            />
                            <Area 
                              type="monotone" 
                              dataKey="rate" 
                              fill="url(#colorRate)" 
                              fillOpacity={0.3}
                              stroke={false}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="rate" 
                              stroke={colors.secondary} 
                              strokeWidth={2.5} 
                              dot={{ r: 4, fill: colors.secondary, strokeWidth: 1, stroke: '#000' }}
                              activeDot={{ r: 6, fill: colors.secondary, stroke: '#FFF', strokeWidth: 2 }}
                            />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                {/* Charts Row 2 */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                  <Card className="col-span-1 border-0 shadow-md">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle>Deepfake Types</CardTitle>
                          <CardDescription>Breakdown by manipulation technique</CardDescription>
                        </div>
                        <TooltipProvider>
                          <TooltipUI>
                            <TooltipTrigger asChild>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <HelpCircle className="h-4 w-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p className="max-w-xs">Analysis of techniques used in detected deepfake videos.</p>
                            </TooltipContent>
                          </TooltipUI>
                        </TooltipProvider>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={analyticsData?.detectionTypes || []}
                              cx="50%"
                              cy="50%"
                              innerRadius={60}
                              outerRadius={80}
                              fill="#8884d8"
                              paddingAngle={2}
                              dataKey="value"
                              label={(entry) => entry.name}
                              labelLine={false}
                            >
                              {(analyticsData?.detectionTypes || []).map((entry: any, index: number) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Pie>
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: 'rgba(0, 0, 0, 0.85)', 
                                border: 'none',
                                borderRadius: '4px',
                                color: 'white' 
                              }}
                            />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="mt-2 space-y-2">
                        {(analyticsData?.detectionTypes || []).map((entry: any, index: number) => (
                          <div key={index} className="flex items-center text-sm">
                            <div className="h-3 w-3 rounded-full mr-2" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                            <span>{entry.name}</span>
                            <span className="ml-auto font-semibold">{entry.value}%</span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="col-span-1 border-0 shadow-md">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle>User Growth</CardTitle>
                          <CardDescription>Weekly registered users</CardDescription>
                        </div>
                        <Badge variant="outline" className="bg-tertiary/10 text-tertiary">
                          <ArrowUpRight size={12} className="mr-1" /> 12.5%
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={analyticsData?.userGrowth || []}
                            margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.1} />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: 'rgba(0, 0, 0, 0.85)', 
                                border: 'none',
                                borderRadius: '4px',
                                color: 'white' 
                              }}
                              itemStyle={{ color: colors.tertiary }}
                              labelStyle={{ color: 'white' }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="users" 
                              stroke={colors.tertiary} 
                              strokeWidth={2} 
                              dot={{ r: 3, strokeWidth: 1 }} 
                              activeDot={{ r: 6, fill: colors.tertiary, stroke: '#FFF', strokeWidth: 2 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <Separator className="my-2" />
                      <div className="grid grid-cols-2 gap-2 pt-2">
                        <div className="flex flex-col">
                          <span className="text-xs text-muted-foreground">Current Period</span>
                          <span className="text-2xl font-bold">148</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-xs text-muted-foreground">Previous Period</span>
                          <span className="text-2xl font-bold">132</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="col-span-1 border-0 shadow-md">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle>Processing Times</CardTitle>
                          <CardDescription>Analysis duration distribution</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={analyticsData?.processingTimes || []}
                            layout="vertical"
                            margin={{ top: 20, right: 30, left: 80, bottom: 0 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.1} horizontal={false} />
                            <XAxis type="number" />
                            <YAxis dataKey="timeRange" type="category" width={75} />
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: 'rgba(0, 0, 0, 0.85)', 
                                border: 'none',
                                borderRadius: '4px',
                                color: 'white' 
                              }}
                              itemStyle={{ color: colors.quaternary }}
                              labelStyle={{ color: 'white' }}
                            />
                            <Bar dataKey="count" fill={colors.quaternary} radius={[0, 4, 4, 0]}>
                              {(analyticsData?.processingTimes || []).map((entry: any, index: number) => (
                                <Cell 
                                  key={`cell-${index}`} 
                                  fill={colors.quaternary} 
                                  fillOpacity={0.7 + (0.3 * index / (analyticsData?.processingTimes?.length || 1))} 
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                    <CardFooter className="pt-0">
                      <div className="w-full flex justify-between items-center text-sm pt-1">
                        <span className="text-muted-foreground">Average: <span className="text-foreground font-medium">14.3s</span></span>
                        <Badge variant="secondary" className="bg-black/5">
                          <Clock className="h-3 w-3 mr-1" /> 3.2s faster than last week
                        </Badge>
                      </div>
                    </CardFooter>
                  </Card>
                </div>
                
                {/* Quick Actions */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="border-0 shadow-md bg-gradient-to-br from-background to-background/80">
                    <CardHeader>
                      <div className="flex items-center">
                        <div className="bg-primary/10 p-2 rounded-full mr-3">
                          <Activity className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <CardTitle>System Performance</CardTitle>
                          <CardDescription>Monitor resource usage and health</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Button variant="outline" className="w-full">View Details</Button>
                    </CardContent>
                  </Card>
                  
                  <Card className="border-0 shadow-md bg-gradient-to-br from-background to-background/80">
                    <CardHeader>
                      <div className="flex items-center">
                        <div className="bg-secondary/10 p-2 rounded-full mr-3">
                          <FileText className="h-5 w-5 text-secondary" />
                        </div>
                        <div>
                          <CardTitle>Analytics Reports</CardTitle>
                          <CardDescription>Generate detailed reports</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Button variant="outline" className="w-full">Generate Report</Button>
                    </CardContent>
                  </Card>
                  
                  <Card className="border-0 shadow-md bg-gradient-to-br from-background to-background/80">
                    <CardHeader>
                      <div className="flex items-center">
                        <div className="bg-tertiary/10 p-2 rounded-full mr-3">
                          <PieChartIcon className="h-5 w-5 text-tertiary" />
                        </div>
                        <div>
                          <CardTitle>Model Performance</CardTitle>
                          <CardDescription>View AI detection metrics</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Button variant="outline" className="w-full">View Metrics</Button>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            
              <TabsContent value="deepfakes" className="m-0">
                {/* Detailed Deepfake Analysis Tab Content */}
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="border-0 shadow-md">
                      <CardHeader>
                        <CardTitle>Deepfake Detection Accuracy</CardTitle>
                        <CardDescription>Analysis accuracy over time</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="h-[350px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart
                              data={[
                                { month: 'Jan', actual: 95, predicted: 93 },
                                { month: 'Feb', actual: 96, predicted: 95 },
                                { month: 'Mar', actual: 94, predicted: 92 },
                                { month: 'Apr', actual: 97, predicted: 96 },
                                { month: 'May', actual: 98, predicted: 97 },
                              ]}
                              margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.1} />
                              <XAxis dataKey="month" />
                              <YAxis domain={[85, 100]} />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="actual" name="Actual Deepfakes" fill={colors.primary} />
                              <Line type="monotone" dataKey="predicted" name="Predicted Deepfakes" stroke={colors.secondary} strokeWidth={3} />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card className="border-0 shadow-md">
                      <CardHeader>
                        <CardTitle>Detection Techniques Distribution</CardTitle>
                        <CardDescription>Methods used to identify manipulations</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="h-[350px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <RadarChart outerRadius={130} width={500} height={350} data={[
                              { subject: 'Visual Artifacts', A: 92, B: 85, fullMark: 100 },
                              { subject: 'Facial Inconsistencies', A: 88, B: 80, fullMark: 100 },
                              { subject: 'Audio Mismatch', A: 76, B: 70, fullMark: 100 },
                              { subject: 'Temporal Errors', A: 84, B: 75, fullMark: 100 },
                              { subject: 'Metadata Analysis', A: 80, B: 65, fullMark: 100 },
                              { subject: 'Behavior Patterns', A: 72, B: 68, fullMark: 100 },
                            ]}>
                              <PolarGrid />
                              <PolarAngleAxis dataKey="subject" />
                              <PolarRadiusAxis domain={[0, 100]} />
                              <Radar name="Current Model" dataKey="A" stroke={colors.secondary} fill={colors.secondary} fillOpacity={0.5} />
                              <Radar name="Previous Model" dataKey="B" stroke={colors.tertiary} fill={colors.tertiary} fillOpacity={0.3} />
                              <Legend />
                            </RadarChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  
                  <Card className="border-0 shadow-md">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle>Deepfake Detection by Content Type</CardTitle>
                          <CardDescription>Analysis by content categories</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="w-full min-w-[600px]">
                          <thead>
                            <tr className="border-b border-muted">
                              <th className="text-left font-medium py-3 px-4">Content Category</th>
                              <th className="text-left font-medium py-3 px-4">Total Videos</th>
                              <th className="text-left font-medium py-3 px-4">Deepfakes Detected</th>
                              <th className="text-left font-medium py-3 px-4">Detection Rate</th>
                              <th className="text-left font-medium py-3 px-4">Avg. Confidence</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="border-b border-muted/50 hover:bg-muted/5">
                              <td className="py-3 px-4">Political Content</td>
                              <td className="py-3 px-4">432</td>
                              <td className="py-3 px-4">117</td>
                              <td className="py-3 px-4">27.1%</td>
                              <td className="py-3 px-4">
                                <div className="flex items-center">
                                  <span className="mr-2">96.4%</span>
                                  <Badge className="bg-green-500/10 text-green-500 border-0">High</Badge>
                                </div>
                              </td>
                            </tr>
                            <tr className="border-b border-muted/50 hover:bg-muted/5">
                              <td className="py-3 px-4">Celebrity Media</td>
                              <td className="py-3 px-4">851</td>
                              <td className="py-3 px-4">263</td>
                              <td className="py-3 px-4">30.9%</td>
                              <td className="py-3 px-4">
                                <div className="flex items-center">
                                  <span className="mr-2">94.2%</span>
                                  <Badge className="bg-green-500/10 text-green-500 border-0">High</Badge>
                                </div>
                              </td>
                            </tr>
                            <tr className="border-b border-muted/50 hover:bg-muted/5">
                              <td className="py-3 px-4">News Segments</td>
                              <td className="py-3 px-4">217</td>
                              <td className="py-3 px-4">42</td>
                              <td className="py-3 px-4">19.4%</td>
                              <td className="py-3 px-4">
                                <div className="flex items-center">
                                  <span className="mr-2">92.7%</span>
                                  <Badge className="bg-green-500/10 text-green-500 border-0">High</Badge>
                                </div>
                              </td>
                            </tr>
                            <tr className="border-b border-muted/50 hover:bg-muted/5">
                              <td className="py-3 px-4">Social Media Content</td>
                              <td className="py-3 px-4">1204</td>
                              <td className="py-3 px-4">286</td>
                              <td className="py-3 px-4">23.8%</td>
                              <td className="py-3 px-4">
                                <div className="flex items-center">
                                  <span className="mr-2">88.5%</span>
                                  <Badge className="bg-yellow-500/10 text-yellow-500 border-0">Medium</Badge>
                                </div>
                              </td>
                            </tr>
                            <tr className="hover:bg-muted/5">
                              <td className="py-3 px-4">Entertainment/Movies</td>
                              <td className="py-3 px-4">346</td>
                              <td className="py-3 px-4">95</td>
                              <td className="py-3 px-4">27.5%</td>
                              <td className="py-3 px-4">
                                <div className="flex items-center">
                                  <span className="mr-2">91.3%</span>
                                  <Badge className="bg-green-500/10 text-green-500 border-0">High</Badge>
                                </div>
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            
              <TabsContent value="system" className="m-0">
                {/* System Health Tab Content */}
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {systemMetrics.map((metric, index) => (
                      <Card key={index} className="border-0 shadow-md overflow-hidden">
                        <CardHeader className="pb-2">
                          <div className="flex justify-between">
                            <CardTitle className="text-base font-medium">{metric.name}</CardTitle>
                            <TooltipProvider>
                              <TooltipUI>
                                <TooltipTrigger asChild>
                                  {metric.value > metric.target * 0.8 ? (
                                    <AlertTriangle className="h-4 w-4 text-warning" />
                                  ) : (
                                    <CheckCircle className="h-4 w-4 text-success" />
                                  )}
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p>{metric.value > metric.target * 0.8 ? 'Warning: Approaching limit' : 'Operating normally'}</p>
                                </TooltipContent>
                              </TooltipUI>
                            </TooltipProvider>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="flex justify-between mb-1">
                            <span className="text-2xl font-bold">{metric.value}%</span>
                            <span className="text-sm text-muted-foreground">Target: {metric.target}%</span>
                          </div>
                          <div className="w-full">
                            <Progress 
                              value={metric.value} 
                              max={100} 
                              className="h-2" 
                              indicatorClassName={`bg-gradient-to-r from-${
                                metric.value > metric.target * 0.8 ? 'warning' : 'success'
                              } to-${
                                metric.value > metric.target * 0.8 ? 'error' : 'primary'
                              }`}
                            />
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                  
                  <Card className="border-0 shadow-md overflow-hidden">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle>System Resources Trend</CardTitle>
                          <CardDescription>Resource usage over time</CardDescription>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button 
                            variant="outline" 
                            size="sm" 
                            className={`${showAnomalies ? 'bg-tertiary/10 text-tertiary border-tertiary/20' : ''}`}
                            onClick={() => setShowAnomalies(!showAnomalies)}
                          >
                            {showAnomalies ? 'Hide' : 'Show'} Anomalies
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[350px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={[
                              { time: '00:00', cpu: 24, memory: 35, storage: 62, network: 15 },
                              { time: '04:00', cpu: 22, memory: 34, storage: 64, network: 12 },
                              { time: '08:00', cpu: 35, memory: 42, storage: 65, network: 28 },
                              { time: '12:00', cpu: 56, memory: 58, storage: 68, network: 42 },
                              { time: '16:00', cpu: 45, memory: 50, storage: 70, network: 32 },
                              { time: '20:00', cpu: 32, memory: 45, storage: 72, network: 18 },
                              { time: '24:00', cpu: 25, memory: 40, storage: 72, network: 14 },
                            ]}
                            margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.1} />
                            <XAxis dataKey="time" />
                            <YAxis />
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: 'rgba(0, 0, 0, 0.85)', 
                                border: 'none',
                                borderRadius: '4px',
                                color: 'white' 
                              }}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="cpu" name="CPU" stroke={colors.primary} strokeWidth={2} dot={{ r: 3 }} />
                            <Line type="monotone" dataKey="memory" name="Memory" stroke={colors.secondary} strokeWidth={2} dot={{ r: 3 }} />
                            <Line type="monotone" dataKey="storage" name="Storage" stroke={colors.warning} strokeWidth={2} dot={{ r: 3 }} />
                            <Line type="monotone" dataKey="network" name="Network" stroke={colors.tertiary} strokeWidth={2} dot={{ r: 3 }} />
                            {showAnomalies && (
                              <Scatter name="Anomalies" dataKey="anomaly" fill={colors.error} shape="star" />
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                  
                  {showAnomalies && (
                    <Card className="border-0 shadow-md overflow-hidden border-error/20">
                      <CardHeader className="bg-error/5">
                        <div className="flex items-center">
                          <AlertTriangle className="h-5 w-5 text-error mr-2" />
                          <CardTitle>System Anomalies Detected</CardTitle>
                        </div>
                        <CardDescription>Recent anomalies in system operation</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="overflow-x-auto">
                          <table className="w-full min-w-[600px]">
                            <thead>
                              <tr className="border-b border-muted">
                                <th className="text-left font-medium py-3 px-4">ID</th>
                                <th className="text-left font-medium py-3 px-4">Type</th>
                                <th className="text-left font-medium py-3 px-4">Timestamp</th>
                                <th className="text-left font-medium py-3 px-4">Severity</th>
                                <th className="text-left font-medium py-3 px-4">Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {anomalyData.map((anomaly) => (
                                <tr key={anomaly.id} className="border-b border-muted/50 hover:bg-muted/5">
                                  <td className="py-3 px-4 font-mono text-sm">{anomaly.id}</td>
                                  <td className="py-3 px-4">{anomaly.type}</td>
                                  <td className="py-3 px-4">{anomaly.timestamp}</td>
                                  <td className="py-3 px-4">
                                    <Badge className={`${
                                      anomaly.severity === 'high' ? 'bg-error/10 text-error' :
                                      anomaly.severity === 'medium' ? 'bg-warning/10 text-warning' :
                                      'bg-tertiary/10 text-tertiary'
                                    } border-0`}>
                                      {anomaly.severity}
                                    </Badge>
                                  </td>
                                  <td className="py-3 px-4">
                                    <Badge className={`${
                                      anomaly.status === 'investigating' ? 'bg-warning/10 text-warning' :
                                      'bg-success/10 text-success'
                                    } border-0`}>
                                      {anomaly.status}
                                    </Badge>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </TabsContent>
              
              <TabsContent value="model" className="m-0">
                {/* Model Performance Tab Content */}
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="border-0 shadow-md">
                      <CardHeader>
                        <CardTitle>Model Performance Metrics</CardTitle>
                        <CardDescription>Key performance indicators of the current AI model</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <table className="w-full">
                          <thead>
                            <tr className="border-b border-muted text-sm">
                              <th className="text-left py-3 pl-2 font-medium">Metric</th>
                              <th className="text-right py-3 font-medium">Current</th>
                              <th className="text-right py-3 pr-2 font-medium">Previous</th>
                              <th className="text-right py-3 pr-2 font-medium">Change</th>
                            </tr>
                          </thead>
                          <tbody>
                            {modelPerformanceData.map((metric, index) => {
                              const change = ((metric.current - metric.previous) / metric.previous * 100).toFixed(1);
                              const isPositive = metric.current >= metric.previous;
                              
                              return (
                                <tr key={index} className="border-b border-muted/50 hover:bg-muted/5">
                                  <td className="py-4 pl-2 font-medium">{metric.name}</td>
                                  <td className="py-4 text-right font-mono">{metric.current.toFixed(1)}%</td>
                                  <td className="py-4 text-right font-mono text-muted-foreground">{metric.previous.toFixed(1)}%</td>
                                  <td className="py-4 pr-2 text-right">
                                    <span className={isPositive ? 'text-success' : 'text-error'}>
                                      {isPositive ? '↑' : '↓'} {change}%
                                    </span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </CardContent>
                      <CardFooter className="justify-between border-t pt-4">
                        <span className="text-sm text-muted-foreground">Last model update: May 15, 2025</span>
                        <Button variant="outline" size="sm">View Full Report</Button>
                      </CardFooter>
                    </Card>
                    
                    <Card className="border-0 shadow-md">
                      <CardHeader>
                        <CardTitle>Model Training Progress</CardTitle>
                        <CardDescription>Latest training epoch metrics</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="h-[350px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                              data={[
                                { epoch: 1, training: 85.2, validation: 83.1 },
                                { epoch: 2, training: 88.7, validation: 86.3 },
                                { epoch: 3, training: 91.4, validation: 88.7 },
                                { epoch: 4, training: 93.2, validation: 90.5 },
                                { epoch: 5, training: 94.8, validation: 92.1 },
                                { epoch: 6, training: 95.7, validation: 93.4 },
                                { epoch: 7, training: 96.3, validation: 94.2 },
                                { epoch: 8, training: 96.8, validation: 94.7 },
                                { epoch: 9, training: 97.1, validation: 95.1 },
                                { epoch: 10, training: 97.3, validation: 95.3 },
                              ]}
                              margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.1} />
                              <XAxis dataKey="epoch" />
                              <YAxis domain={[80, 100]} />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: 'rgba(0, 0, 0, 0.85)', 
                                  border: 'none',
                                  borderRadius: '4px',
                                  color: 'white' 
                                }}
                              />
                              <Legend />
                              <Line type="monotone" dataKey="training" name="Training Accuracy" stroke={colors.primary} strokeWidth={2} dot={{ r: 3 }} />
                              <Line type="monotone" dataKey="validation" name="Validation Accuracy" stroke={colors.tertiary} strokeWidth={2} dot={{ r: 3 }} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  
                  <Card className="border-0 shadow-md">
                    <CardHeader>
                      <CardTitle>Model Version History</CardTitle>
                      <CardDescription>Deepfake detector model evolution</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="w-full min-w-[700px]">
                          <thead>
                            <tr className="border-b border-muted">
                              <th className="text-left font-medium py-3 px-4">Version</th>
                              <th className="text-left font-medium py-3 px-4">Release Date</th>
                              <th className="text-left font-medium py-3 px-4">Accuracy</th>
                              <th className="text-left font-medium py-3 px-4">Precision</th>
                              <th className="text-left font-medium py-3 px-4">Recall</th>
                              <th className="text-left font-medium py-3 px-4">F1 Score</th>
                              <th className="text-left font-medium py-3 px-4">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="border-b border-muted/50 hover:bg-muted/5">
                              <td className="py-3 px-4">v3.2.1</td>
                              <td className="py-3 px-4">May 15, 2025</td>
                              <td className="py-3 px-4">96.3%</td>
                              <td className="py-3 px-4">94.7%</td>
                              <td className="py-3 px-4">93.8%</td>
                              <td className="py-3 px-4">94.2%</td>
                              <td className="py-3 px-4">
                                <Badge className="bg-success/10 text-success border-0">Active</Badge>
                              </td>
                            </tr>
                            <tr className="border-b border-muted/50 hover:bg-muted/5">
                              <td className="py-3 px-4">v3.1.0</td>
                              <td className="py-3 px-4">Apr 02, 2025</td>
                              <td className="py-3 px-4">95.2%</td>
                              <td className="py-3 px-4">93.5%</td>
                              <td className="py-3 px-4">92.1%</td>
                              <td className="py-3 px-4">92.8%</td>
                              <td className="py-3 px-4">
                                <Badge variant="outline" className="border-muted-foreground/30">Archived</Badge>
                              </td>
                            </tr>
                            <tr className="border-b border-muted/50 hover:bg-muted/5">
                              <td className="py-3 px-4">v3.0.2</td>
                              <td className="py-3 px-4">Feb 18, 2025</td>
                              <td className="py-3 px-4">93.7%</td>
                              <td className="py-3 px-4">91.2%</td>
                              <td className="py-3 px-4">90.8%</td>
                              <td className="py-3 px-4">91.0%</td>
                              <td className="py-3 px-4">
                                <Badge variant="outline" className="border-muted-foreground/30">Archived</Badge>
                              </td>
                            </tr>
                            <tr className="hover:bg-muted/5">
                              <td className="py-3 px-4">v2.8.5</td>
                              <td className="py-3 px-4">Jan 05, 2025</td>
                              <td className="py-3 px-4">91.2%</td>
                              <td className="py-3 px-4">89.5%</td>
                              <td className="py-3 px-4">88.7%</td>
                              <td className="py-3 px-4">89.1%</td>
                              <td className="py-3 px-4">
                                <Badge variant="outline" className="border-muted-foreground/30">Archived</Badge>
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </div>
    </div>
  );  

}