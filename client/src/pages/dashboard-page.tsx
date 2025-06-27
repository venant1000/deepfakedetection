import { useState } from "react";
import Sidebar from "@/components/layout/sidebar";
import StatsOverview from "@/components/dashboard/stats-overview";

import RecentAnalyses from "@/components/dashboard/recent-analyses";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { RefreshCw, Loader2, Download, FileText, FileSpreadsheet } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiRequest } from "@/lib/queryClient";

export default function DashboardPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [downloading, setDownloading] = useState<string | null>(null);

  // Format date for display
  const formatLastUpdated = () => {
    return lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Handle manual refresh - reload the entire page to fetch fresh data
  const handleRefresh = async () => {
    setRefreshing(true);
    
    try {
      // Show immediate feedback
      toast({
        title: "Refreshing dashboard",
        description: "Reloading all system data...",
      });
      
      // Reload the entire page to fetch fresh data from all components
      window.location.reload();
    } catch (error) {
      toast({
        title: "Refresh failed",
        description: "Unable to refresh dashboard data. Please try again.",
        variant: "destructive",
      });
      setRefreshing(false);
    }
  };

  // Handle file downloads
  const handleDownload = async (downloadType: string, fileName: string) => {
    setDownloading(downloadType);
    
    try {
      // Make the download request
      const response = await fetch(`/api/download/${downloadType}`, {
        method: 'GET',
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Download failed');
      }

      // Get the blob data
      const blob = await response.blob();
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      
      // Clean up
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      toast({
        title: "Download successful",
        description: `${fileName} has been downloaded.`,
      });
    } catch (error) {
      console.error('Download error:', error);
      toast({
        title: "Download failed",
        description: error instanceof Error ? error.message : "Unable to download file",
        variant: "destructive",
      });
    } finally {
      setDownloading(null);
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
        
        {/* Download Section */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Download className="h-5 w-5" />
              Download Your Data
            </CardTitle>
            <CardDescription>
              Export your analysis data and reports for backup or external use
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Button
                onClick={() => handleDownload('all-analyses', `all-analyses-${user?.username}.json`)}
                disabled={downloading === 'all-analyses'}
                className="flex items-center gap-2 h-12"
              >
                {downloading === 'all-analyses' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <FileText className="h-4 w-4" />
                )}
                Download All Analyses (JSON)
              </Button>
              
              <Button
                onClick={() => handleDownload('analyses-csv', `analyses-${user?.username}.csv`)}
                disabled={downloading === 'analyses-csv'}
                variant="outline"
                className="flex items-center gap-2 h-12"
              >
                {downloading === 'analyses-csv' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <FileSpreadsheet className="h-4 w-4" />
                )}
                Download Analyses (CSV)
              </Button>
              
              {user?.username?.includes('admin') && (
                <Button
                  onClick={() => handleDownload('system-logs', 'system-logs.json')}
                  disabled={downloading === 'system-logs'}
                  variant="secondary"
                  className="flex items-center gap-2 h-12 md:col-span-2"
                >
                  {downloading === 'system-logs' ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <FileText className="h-4 w-4" />
                  )}
                  Download System Logs (Admin Only)
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
        
        <RecentAnalyses />
      </div>
    </div>
  );
}
