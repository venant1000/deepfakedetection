import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Loader2, RefreshCw } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import AnalyticsDashboard from "@/components/admin/analytics-dashboard-live";

export default function AdminAnalytics() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(true);
  const [analyticsData, setAnalyticsData] = useState<any>(null);
  
  // Format date for display
  const formatLastUpdated = () => {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Fetch analytics data - with debounce to prevent multiple requests
  const fetchAnalytics = async () => {
    try {
      setIsLoading(true);
      
      const response = await fetch("/api/admin/stats", {
        credentials: "include",
        cache: "no-store" // Prevent caching
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

      // Get data from the API
      const data = await response.json();
      
      // Set data immediately without delay
      setAnalyticsData(data);
      setIsLoading(false);
      
    } catch (error) {
      console.error("Error fetching analytics:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to load analytics data",
        variant: "destructive",
      });
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);
  
  // Handle manual refresh
  const handleRefresh = () => {
    fetchAnalytics();
  };

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
                <span>Last updated: {formatLastUpdated()}</span>
              </div>
            </div>
            
            <div className="flex flex-wrap items-center gap-2">
              <Button 
                size="icon" 
                variant="ghost" 
                onClick={handleRefresh} 
                className="h-8 w-8"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
          
          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
            <Card className="shadow-md">
              <CardHeader>
                <CardDescription>Total Users</CardDescription>
                <CardTitle className="text-3xl">{analyticsData?.summary?.totalUsers || 0}</CardTitle>
              </CardHeader>
            </Card>

            <Card className="shadow-md">
              <CardHeader>
                <CardDescription>Videos Analyzed</CardDescription>
                <CardTitle className="text-3xl">{analyticsData?.summary?.videoCount || 0}</CardTitle>
              </CardHeader>
            </Card>

            <Card className="shadow-md">
              <CardHeader>
                <CardDescription>Deepfakes Detected</CardDescription>
                <CardTitle className="text-3xl">{analyticsData?.summary?.deepfakeCount || 0}</CardTitle>
              </CardHeader>
            </Card>

            <Card className="shadow-md">
              <CardHeader>
                <CardDescription>System Health</CardDescription>
                <CardTitle className="text-3xl">{analyticsData?.summary?.systemHealth || 0}%</CardTitle>
              </CardHeader>
            </Card>
          </div>
          
          {/* Analytics Dashboard with real-time data */}
          <AnalyticsDashboard analyticsData={analyticsData} />
        </div>
      </div>
    </div>
  );
}