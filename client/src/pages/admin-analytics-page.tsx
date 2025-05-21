import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";
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
} from "recharts";

export default function AdminAnalyticsPage() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(true);
  const [timeRange, setTimeRange] = useState("last30days");
  const [analyticsData, setAnalyticsData] = useState<any>(null);
  
  // Fetch analytics data
  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
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

        const data = await response.json();
        
        // In a real app, this would be provided by the API
        // For this demo, we'll simulate it
        setAnalyticsData({
          summary: {
            totalUsers: data.totalUsers || 24,
            videoCount: data.videosAnalyzed || 146,
            deepfakesDetected: data.deepfakesDetected || 67,
            systemHealth: data.systemHealth || 98.7
          },
          dailyUploads: [
            { date: "May 15", count: 12 },
            { date: "May 16", count: 8 },
            { date: "May 17", count: 17 },
            { date: "May 18", count: 15 },
            { date: "May 19", count: 21 },
            { date: "May 20", count: 19 },
            { date: "May 21", count: 23 }
          ],
          detectionRates: [
            { date: "May 15", rate: 43.2 },
            { date: "May 16", rate: 38.5 },
            { date: "May 17", rate: 46.7 },
            { date: "May 18", rate: 41.3 },
            { date: "May 19", rate: 45.8 },
            { date: "May 20", rate: 50.2 },
            { date: "May 21", rate: 47.6 }
          ],
          detectionTypes: [
            { name: "Facial Manipulation", value: 38 },
            { name: "Voice Synthesis", value: 17 },
            { name: "Body Movements", value: 8 },
            { name: "Background Alterations", value: 4 }
          ],
          userGrowth: [
            { date: "Apr 21", users: 8 },
            { date: "Apr 28", users: 12 },
            { date: "May 5", users: 15 },
            { date: "May 12", users: 19 },
            { date: "May 19", users: 24 }
          ],
          processingTimes: [
            { timeRange: "<30s", count: 36 },
            { timeRange: "30s-1m", count: 52 },
            { timeRange: "1m-2m", count: 38 },
            { timeRange: "2m-5m", count: 15 },
            { timeRange: ">5m", count: 5 }
          ]
        });
      } catch (error) {
        toast({
          title: "Error",
          description: error instanceof Error ? error.message : "Failed to load analytics data",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnalytics();
  }, [toast, navigate]);

  // Chart colors
  const colors = {
    primary: "#00ff88",
    secondary: "#7000ff",
    tertiary: "#00a3ff",
    quaternary: "#ff3e66",
    neutral: "#888888"
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
      
      <div className="flex-1 ml-20 md:ml-64 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
            <h1 className="text-3xl font-bold">Analytics Dashboard</h1>
            
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
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Total Users</CardDescription>
                <CardTitle className="text-3xl">{analyticsData.summary.totalUsers}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><path d="m18 15-6-6-6 6"/></svg>
                  <span className="text-green-500">+4</span>
                  <span className="ml-1">since last week</span>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Videos Analyzed</CardDescription>
                <CardTitle className="text-3xl">{analyticsData.summary.videoCount}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><path d="m18 15-6-6-6 6"/></svg>
                  <span className="text-green-500">+23</span>
                  <span className="ml-1">since last week</span>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Deepfakes Detected</CardDescription>
                <CardTitle className="text-3xl">{analyticsData.summary.deepfakesDetected}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><path d="m18 15-6-6-6 6"/></svg>
                  <span className="text-green-500">+11</span>
                  <span className="ml-1">since last week</span>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>System Health</CardDescription>
                <CardTitle className="text-3xl">{analyticsData.summary.systemHealth}%</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-muted-foreground flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                  <span>System operating normally</span>
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Charts Row 1 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle>Daily Video Uploads</CardTitle>
                <CardDescription>Number of videos uploaded per day</CardDescription>
              </CardHeader>
              <CardContent className="pt-2">
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={analyticsData.dailyUploads}
                      margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill={colors.primary} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle>Detection Rate Trend</CardTitle>
                <CardDescription>Percentage of videos identified as deepfakes</CardDescription>
              </CardHeader>
              <CardContent className="pt-2">
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={analyticsData.detectionRates}
                      margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="rate" stroke={colors.secondary} strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Charts Row 2 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle>Deepfake Types</CardTitle>
                <CardDescription>Breakdown by manipulation technique</CardDescription>
              </CardHeader>
              <CardContent className="pt-2">
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={analyticsData.detectionTypes}
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
                        {analyticsData.detectionTypes.map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle>User Growth</CardTitle>
                <CardDescription>Weekly registered users</CardDescription>
              </CardHeader>
              <CardContent className="pt-2">
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={analyticsData.userGrowth}
                      margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="users" stroke={colors.tertiary} strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle>Processing Times</CardTitle>
                <CardDescription>Analysis duration distribution</CardDescription>
              </CardHeader>
              <CardContent className="pt-2">
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={analyticsData.processingTimes}
                      layout="vertical"
                      margin={{ top: 20, right: 30, left: 40, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="timeRange" type="category" />
                      <Tooltip />
                      <Bar dataKey="count" fill={colors.quaternary} radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>System Performance</CardTitle>
                <CardDescription>Monitor system resource usage</CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">View Details</Button>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Analytics Reports</CardTitle>
                <CardDescription>Generate detailed analytics reports</CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">Generate Report</Button>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Model Performance</CardTitle>
                <CardDescription>Evaluate AI detection accuracy</CardDescription>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">View Metrics</Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}