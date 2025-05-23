import { useState } from "react";
import Sidebar from "@/components/layout/sidebar";
import StatsOverview from "@/components/dashboard/stats-overview";
import UploadSection from "@/components/dashboard/upload-section";
import RecentAnalyses from "@/components/dashboard/recent-analyses";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { RefreshCw, Loader2 } from "lucide-react";

export default function DashboardPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Format date for display
  const formatLastUpdated = () => {
    return lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Handle manual refresh
  const handleRefresh = async () => {
    setRefreshing(true);
    
    try {
      // Simulate refresh delay and update timestamp
      await new Promise(resolve => setTimeout(resolve, 1000));
      setLastUpdated(new Date());
      
      toast({
        title: "Dashboard refreshed",
        description: "Your dashboard data has been updated.",
      });
    } catch (error) {
      toast({
        title: "Refresh failed",
        description: "Unable to refresh dashboard data. Please try again.",
        variant: "destructive",
      });
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Dashboard Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-2xl font-bold">Dashboard</h1>
            <div className="flex items-center gap-4">
              <p className="text-muted-foreground">Welcome back, {user?.username}</p>
              <div className="flex items-center text-sm text-muted-foreground">
                <span>Last updated: {formatLastUpdated()}</span>
                {refreshing && <Loader2 className="h-3 w-3 ml-2 animate-spin" />}
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <Button 
              size="icon" 
              variant="ghost" 
              onClick={handleRefresh} 
              disabled={refreshing}
              className="h-8 w-8"
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            </Button>
            
            <button className="p-2 rounded-lg glass-dark text-muted-foreground hover:text-white transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>
            </button>
            
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
              </div>
              <div className="hidden md:block">
                <p className="font-medium">{user?.username}</p>
                <p className="text-sm text-muted-foreground">Active User</p>
              </div>
            </div>
          </div>
        </div>
        
        <StatsOverview />
        <UploadSection />
        <RecentAnalyses />
      </div>
    </div>
  );
}
