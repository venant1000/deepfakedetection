import { useParams, useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import AnalysisSummary from "@/components/analysis/analysis-summary";
import VideoAnalysis from "@/components/analysis/video-analysis";
import DeepfakeDetails from "@/components/analysis/deepfake-details";
import { useQuery } from "@tanstack/react-query";
import { VideoAnalysisResult } from "@shared/schema";
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function AnalysisPage() {
  const { id } = useParams();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [analysisData, setAnalysisData] = useState<any>(null);
  
  // Fetch the real video analysis data from the API
  const { data: videoAnalysis, isLoading, error } = useQuery<VideoAnalysisResult>({
    queryKey: [`/api/videos/${id}`],
    enabled: !!id
  });
  
  useEffect(() => {
    if (videoAnalysis) {
      // Create a properly formatted analysis object directly from the API data
      const formattedAnalysis = {
        id: videoAnalysis.id,
        fileName: videoAnalysis.fileName,
        duration: "2:30", // We'll use a fixed duration until we have actual video metadata
        date: new Date(videoAnalysis.uploadDate).toLocaleDateString("en-US", {
          year: 'numeric',
          month: 'long',
          day: 'numeric'
        }),
        isDeepfake: videoAnalysis.analysis.isDeepfake,
        confidence: videoAnalysis.analysis.confidence,
        
        // Use the actual issues from the API response if available
        issues: videoAnalysis.analysis.issues && videoAnalysis.analysis.issues.length > 0 
          ? videoAnalysis.analysis.issues 
          : [
              {
                type: videoAnalysis.analysis.isDeepfake ? "error" : "info",
                text: videoAnalysis.analysis.isDeepfake 
                  ? "Potential deepfake indicators detected" 
                  : "No significant manipulation detected"
              }
            ],
        
        // Use the actual findings from the API response if available
        findings: videoAnalysis.analysis.findings && videoAnalysis.analysis.findings.length > 0
          ? videoAnalysis.analysis.findings
          : [
              {
                title: videoAnalysis.analysis.isDeepfake ? "AI-Generated Content Detected" : "Authentic Content",
                icon: videoAnalysis.analysis.isDeepfake ? "alert-triangle" : "check-circle",
                severity: videoAnalysis.analysis.isDeepfake ? "high" : "low",
                timespan: "throughout video",
                description: videoAnalysis.analysis.isDeepfake 
                  ? `This video shows signs of manipulation with ${videoAnalysis.analysis.confidence}% confidence.` 
                  : `This video appears to be authentic with ${videoAnalysis.analysis.confidence}% confidence.`
              }
            ],
        
        // Use the actual timeline markers from the API response if available
        timeline: videoAnalysis.analysis.timeline && videoAnalysis.analysis.timeline.length > 0
          ? videoAnalysis.analysis.timeline
          : [
              { 
                position: 50, 
                tooltip: videoAnalysis.analysis.isDeepfake ? "Potential manipulation detected" : "No issues detected", 
                type: (videoAnalysis.analysis.isDeepfake ? "warning" : "normal") as "normal" | "warning" | "danger" 
              }
            ]
      };
      
      setAnalysisData(formattedAnalysis);
    }
  }, [videoAnalysis]);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-96">
            <Loader2 className="h-12 w-12 animate-spin mb-4" />
            <p className="text-lg">Loading analysis data...</p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-96">
            <div className="p-8 glass rounded-xl text-center max-w-lg">
              <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-4 text-red-500">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              <h2 className="text-xl font-bold mb-2">Error Loading Analysis</h2>
              <p className="text-muted-foreground mb-4">
                We couldn't load the analysis data for this video. It may have been deleted or you might not have permission to view it.
              </p>
              <button 
                onClick={() => navigate("/dashboard")}
                className="py-2 px-4 rounded-lg bg-primary text-black font-medium hover:opacity-90 transition-all"
              >
                Return to Dashboard
              </button>
            </div>
          </div>
        ) : analysisData ? (
          <>
            {/* Analysis Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <button 
                    onClick={() => navigate("/dashboard")}
                    className="text-muted-foreground hover:text-white transition-colors"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 19-7-7 7-7"/><path d="M19 12H5"/></svg>
                  </button>
                  <h1 className="text-2xl font-bold">Analysis Results</h1>
                </div>
                <p className="text-muted-foreground">{analysisData.fileName} • {analysisData.duration} minutes • Analyzed {analysisData.date}</p>
              </div>
              
              <div className="flex items-center gap-3">
                <button 
                  className="py-2 px-4 rounded-lg glass-dark text-white hover:bg-muted transition-colors flex items-center gap-2"
                  onClick={() => {
                    try {
                      navigator.clipboard.writeText(window.location.href);
                      toast({
                        title: "Link copied",
                        description: "You can now share this analysis with others"
                      });
                    } catch (err) {
                      toast({
                        title: "Error copying link",
                        description: "Please try again or copy the URL manually",
                        variant: "destructive"
                      });
                    }
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><path d="m16 6-4-4-4 4"/><path d="M12 2v13"/></svg>
                  <span>Share</span>
                </button>
                <button 
                  className="py-2 px-4 rounded-lg bg-primary text-black font-medium hover:opacity-90 transition-all flex items-center gap-2"
                  onClick={() => {
                    if (videoAnalysis) {
                      const blob = new Blob([JSON.stringify(videoAnalysis, null, 2)], { type: "application/json" });
                      const url = URL.createObjectURL(blob);
                      const link = document.createElement("a");
                      link.setAttribute("href", url);
                      link.setAttribute("download", `analysis-${videoAnalysis.id}.json`);
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                      
                      toast({
                        title: "Download successful",
                        description: `Analysis report for ${videoAnalysis.fileName} has been downloaded.`
                      });
                    }
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                  <span>Download Report</span>
                </button>
              </div>
            </div>
            
            <AnalysisSummary analysis={analysisData} />
            <VideoAnalysis analysis={analysisData} />
            <DeepfakeDetails findings={analysisData.findings} />
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-96">
            <div className="p-8 glass rounded-xl text-center max-w-lg">
              <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-4 text-yellow-500">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              <h2 className="text-xl font-bold mb-2">No Analysis Data</h2>
              <p className="text-muted-foreground mb-4">
                No analysis data was found for this video. Please return to the dashboard and try again.
              </p>
              <button 
                onClick={() => navigate("/dashboard")}
                className="py-2 px-4 rounded-lg bg-primary text-black font-medium hover:opacity-90 transition-all"
              >
                Return to Dashboard
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
