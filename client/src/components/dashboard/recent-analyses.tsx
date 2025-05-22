import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { useQuery } from "@tanstack/react-query";
import { VideoAnalysisResult } from "@shared/schema";
import { Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Link } from "wouter";

export default function RecentAnalyses() {
  const [, navigate] = useLocation();
  const { user } = useAuth();
  const { toast } = useToast();
  
  // Fetch real video analyses from the API
  const { data: videoAnalyses, isLoading, error } = useQuery<VideoAnalysisResult[]>({
    queryKey: ["/api/videos"],
    enabled: !!user
  });
  
  // Format a date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    
    // Today
    if (date.toDateString() === now.toDateString()) {
      return `Today, ${date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}`;
    }
    // Yesterday
    else if (date.toDateString() === yesterday.toDateString()) {
      return `Yesterday, ${date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}`;
    }
    // Other dates
    else {
      return date.toLocaleDateString('en-US', { 
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit'
      });
    }
  };
  
  // Get most recent analyses (up to 5)
  const recentAnalyses = videoAnalyses 
    ? [...videoAnalyses]
        .sort((a, b) => new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime())
        .slice(0, 5)
    : [];
  
  const getStatusColor = (isDeepfake: boolean, confidence: number) => {
    if (isDeepfake) {
      return 'bg-[#ff3366]/20 text-[#ff3366]';
    } else if (confidence > 80) {
      return 'bg-primary/20 text-primary';
    } else {
      return 'bg-[#ffbb00]/20 text-[#ffbb00]';
    }
  };
  
  const getStatusText = (isDeepfake: boolean, confidence: number) => {
    if (isDeepfake) {
      return 'Deepfake';
    } else if (confidence > 80) {
      return 'Authentic';
    } else {
      return 'Suspicious';
    }
  };
  
  return (
    <div className="glass rounded-xl p-6 mb-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold">Recent Analyses</h2>
        <Link href="/history" className="text-primary text-sm hover:underline">View All</Link>
      </div>
      
      <div className="overflow-x-auto">
        {isLoading ? (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <span className="ml-2 text-muted-foreground">Loading analyses...</span>
          </div>
        ) : error ? (
          <div className="text-center py-8">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-2 text-red-500">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12" y2="16"/>
            </svg>
            <p className="text-muted-foreground">There was an error loading your analyses.</p>
            <Button variant="outline" size="sm" className="mt-4" onClick={() => window.location.reload()}>
              Retry
            </Button>
          </div>
        ) : recentAnalyses.length === 0 ? (
          <div className="text-center py-8">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-2 text-muted-foreground">
              <rect width="18" height="18" x="3" y="3" rx="2"/>
              <path d="M7 3v18"/>
              <path d="M3 7h18"/>
            </svg>
            <p className="text-muted-foreground">You haven't analyzed any videos yet.</p>
            <Button variant="outline" size="sm" className="mt-4" onClick={() => navigate("/upload")}>
              Upload Your First Video
            </Button>
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-muted">
                <th className="pb-3 text-left text-muted-foreground font-medium">File Name</th>
                <th className="pb-3 text-left text-muted-foreground font-medium">Date</th>
                <th className="pb-3 text-left text-muted-foreground font-medium">Size</th>
                <th className="pb-3 text-left text-muted-foreground font-medium">Result</th>
                <th className="pb-3 text-left text-muted-foreground font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {recentAnalyses.map((analysis) => (
                <tr key={analysis.id} className="border-b border-muted">
                  <td className="py-4">
                    <div className="flex items-center gap-3">
                      <div className="h-9 w-9 rounded bg-muted flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground"><path d="m22 8-6-6H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M10 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/><path d="m22 16-5.23-5.23a1 1 0 0 0-1.41 0L12 14.12l-1.36-1.36a1 1 0 0 0-1.41 0L2 20"/></svg>
                      </div>
                      <span>{analysis.fileName}</span>
                    </div>
                  </td>
                  <td className="py-4 text-muted-foreground">{formatDate(analysis.uploadDate)}</td>
                  <td className="py-4 text-muted-foreground">{analysis.fileSize ? `${(analysis.fileSize / 1024).toFixed(1)} MB` : "N/A"}</td>
                  <td className="py-4">
                    <span className={`py-1 px-3 rounded-full text-sm ${getStatusColor(analysis.analysis.isDeepfake, analysis.analysis.confidence)}`}>
                      {getStatusText(analysis.analysis.isDeepfake, analysis.analysis.confidence)} ({analysis.analysis.confidence}%)
                    </span>
                  </td>
                  <td className="py-4">
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-muted-foreground hover:text-foreground transition-colors"
                        onClick={() => navigate(`/analysis/${analysis.id}`)}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-muted-foreground hover:text-foreground transition-colors"
                        onClick={() => {
                          // Download the analysis data
                          const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: "application/json" });
                          const url = URL.createObjectURL(blob);
                          const link = document.createElement("a");
                          link.setAttribute("href", url);
                          link.setAttribute("download", `analysis-${analysis.id}.json`);
                          document.body.appendChild(link);
                          link.click();
                          document.body.removeChild(link);
                          
                          toast({
                            title: "Download successful",
                            description: `Analysis data for ${analysis.fileName} has been downloaded.`
                          });
                        }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
