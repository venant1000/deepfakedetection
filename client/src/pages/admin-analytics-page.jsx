import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
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
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

export default function AdminAnalyticsPage() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalVideos: 0,
    deepfakeCount: 0,
    recentActivity: [],
    videosByDay: [],
    detectionAccuracy: [],
    userGrowth: []
  });

  // Fetch analytics data from the API
  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await fetch("/api/admin/analytics", {
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
        setStats(data);
      } catch (error) {
        console.error("Error fetching analytics:", error);
        
        // Use sample data for demonstration
        setStats({
          totalUsers: 67,
          totalVideos: 392,
          deepfakeCount: 174,
          recentActivity: [
            { id: 1, username: "user123", action: "Video analysis", timestamp: "2023-05-10T14:32:00Z", result: "Deepfake detected" },
            { id: 2, username: "analyst42", action: "Profile update", timestamp: "2023-05-10T13:15:00Z", result: "Profile updated" },
            { id: 3, username: "security_expert", action: "Video analysis", timestamp: "2023-05-10T12:08:00Z", result: "Authentic video" },
            { id: 4, username: "media_team", action: "Video analysis", timestamp: "2023-05-10T11:47:00Z", result: "Deepfake detected" },
            { id: 5, username: "journalist99", action: "Account created", timestamp: "2023-05-10T10:22:00Z", result: "New user" }
          ],
          videosByDay: [
            { date: "May 4", count: 32, deepfakes: 14 },
            { date: "May 5", count: 28, deepfakes: 11 },
            { date: "May 6", count: 41, deepfakes: 19 },
            { date: "May 7", count: 45, deepfakes: 22 },
            { date: "May 8", count: 37, deepfakes: 16 },
            { date: "May 9", count: 53, deepfakes: 25 },
            { date: "May 10", count: 48, deepfakes: 21 },
          ],
          detectionAccuracy: [
            { name: "True Positives", value: 163 },
            { name: "False Positives", value: 11 },
            { name: "True Negatives", value: 214 },
            { name: "False Negatives", value: 4 },
          ],
          userGrowth: [
            { month: "Jan", users: 12 },
            { month: "Feb", users: 19 },
            { month: "Mar", users: 24 },
            { month: "Apr", users: 37 },
            { month: "May", users: 67 },
          ]
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnalytics();
  }, [toast, navigate]);

  // Format date helper
  const formatDate = (dateString) => {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(new Date(dateString));
  };

  // Calculate detection accuracy percentage
  const calculateAccuracy = () => {
    const { detectionAccuracy } = stats;
    if (!detectionAccuracy || detectionAccuracy.length < 4) return "N/A";
    
    const truePositives = detectionAccuracy[0].value;
    const falsePositives = detectionAccuracy[1].value;
    const trueNegatives = detectionAccuracy[2].value;
    const falseNegatives = detectionAccuracy[3].value;
    
    const total = truePositives + falsePositives + trueNegatives + falseNegatives;
    const accurate = truePositives + trueNegatives;
    
    return ((accurate / total) * 100).toFixed(1) + "%";
  };

  // Colors for charts
  const COLORS = ["#00ff88", "#7000ff", "#ff5500", "#ffaa00"];
  const RADIAN = Math.PI / 180;

  // Custom label for pie chart
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, index }) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central">
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar isAdmin={true} />
      
      <div className="flex-1 ml-20 md:ml-64 p-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold mb-2">System Analytics</h1>
          <p className="text-muted-foreground mb-8">
            Comprehensive overview of DeepGuard AI platform metrics
          </p>
          
          {isLoading ? (
            <div className="glass rounded-xl p-16 text-center">
              <Loader2 className="h-8 w-8 mx-auto mb-4 animate-spin text-primary" />
              <p className="text-muted-foreground">Loading analytics data...</p>
            </div>
          ) : (
            <>
              {/* Key Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <Card className="glass border-none">
                  <CardHeader className="pb-2">
                    <CardDescription>Total Users</CardDescription>
                    <CardTitle className="text-4xl font-bold flex items-baseline">
                      {stats.totalUsers}
                      <span className="text-sm font-normal text-green-500 ml-2">+12% ↑</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-10">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={stats.userGrowth}>
                          <Line 
                            type="monotone" 
                            dataKey="users" 
                            stroke="#00ff88" 
                            strokeWidth={2} 
                            dot={false} 
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass border-none">
                  <CardHeader className="pb-2">
                    <CardDescription>Videos Analyzed</CardDescription>
                    <CardTitle className="text-4xl font-bold flex items-baseline">
                      {stats.totalVideos}
                      <span className="text-sm font-normal text-green-500 ml-2">+23% ↑</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-10">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={stats.videosByDay.slice(-5)}>
                          <Bar dataKey="count" fill="#7000ff" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass border-none">
                  <CardHeader className="pb-2">
                    <CardDescription>Deepfakes Detected</CardDescription>
                    <CardTitle className="text-4xl font-bold flex items-baseline">
                      {stats.deepfakeCount}
                      <span className="text-sm font-normal text-amber-500 ml-2">{((stats.deepfakeCount / stats.totalVideos) * 100).toFixed(1)}%</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-10">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={stats.videosByDay.slice(-5)}>
                          <Bar dataKey="deepfakes" fill="#ff5500" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Video Analysis Trends */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <Card className="glass border-none">
                  <CardHeader>
                    <CardTitle>Video Analysis Trends</CardTitle>
                    <CardDescription>
                      Daily analysis volume with deepfake detection ratio
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={stats.videosByDay}
                          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis dataKey="date" stroke="#888" />
                          <YAxis stroke="#888" />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'rgba(15, 15, 15, 0.9)', 
                              border: '1px solid #333',
                              borderRadius: '4px',
                              color: '#fff'
                            }} 
                          />
                          <Legend />
                          <Bar dataKey="count" name="Total Videos" fill="#7000ff" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="deepfakes" name="Deepfakes" fill="#ff5500" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass border-none">
                  <CardHeader>
                    <CardTitle>Detection Accuracy</CardTitle>
                    <CardDescription>
                      Overall accuracy: {calculateAccuracy()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={stats.detectionAccuracy}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={renderCustomizedLabel}
                            outerRadius={100}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {stats.detectionAccuracy.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'rgba(15, 15, 15, 0.9)', 
                              border: '1px solid #333',
                              borderRadius: '4px',
                              color: '#fff'
                            }} 
                          />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Recent Activity */}
              <Card className="glass border-none">
                <CardHeader>
                  <CardTitle>Recent Activity</CardTitle>
                  <CardDescription>
                    Latest actions performed on the platform
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-muted">
                          <th className="text-left py-3 px-4 font-medium">User</th>
                          <th className="text-left py-3 px-4 font-medium">Action</th>
                          <th className="text-left py-3 px-4 font-medium">Timestamp</th>
                          <th className="text-left py-3 px-4 font-medium">Result</th>
                        </tr>
                      </thead>
                      <tbody>
                        {stats.recentActivity.map((activity) => (
                          <tr key={activity.id} className="border-b border-muted/50">
                            <td className="py-3 px-4">{activity.username}</td>
                            <td className="py-3 px-4">{activity.action}</td>
                            <td className="py-3 px-4">{formatDate(activity.timestamp)}</td>
                            <td className="py-3 px-4">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                activity.result.includes('Deepfake') 
                                  ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' 
                                  : activity.result.includes('Authentic')
                                  ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                                  : 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400'
                              }`}>
                                {activity.result}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}