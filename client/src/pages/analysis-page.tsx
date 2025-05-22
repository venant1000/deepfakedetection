import { useParams, useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import AnalysisSummary from "@/components/analysis/analysis-summary";
import VideoAnalysis from "@/components/analysis/video-analysis";
import DeepfakeDetails from "@/components/analysis/deepfake-details";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { VideoAnalysisResult } from "@shared/schema";

export default function AnalysisPage() {
  const { id } = useParams();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [isDownloading, setIsDownloading] = useState(false);
  const [isSharing, setIsSharing] = useState(false);

  // Fetch the analysis data from the API
  const { data: analysis, isLoading, error } = useQuery<VideoAnalysisResult>({
    queryKey: ['/api/videos', id],
    enabled: !!id
  });

  // Handle errors
  useEffect(() => {
    if (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
      toast({ 
        title: "Error loading analysis",
        description: `We couldn't retrieve this analysis: ${errorMessage}`,
        variant: "destructive"
      });
    }
  }, [error, toast]);

  // Fallback if no data
  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading analysis data...</p>
        </div>
      </div>
    );
  }

  // Handle missing data error
  if (!analysis) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-center max-w-md">
          <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground"><circle cx="12" cy="12" r="10" /><path d="M12 8v4" /><path d="M12 16h.01" /></svg>
          <h2 className="text-2xl font-bold">Analysis Not Found</h2>
          <p className="text-muted-foreground">The analysis you're looking for doesn't exist or you may not have permission to view it.</p>
          <Button onClick={() => navigate("/dashboard")}>Return to Dashboard</Button>
        </div>
      </div>
    );
  }

  // Format duration for display
  const formatDuration = (seconds: number) => {
    if (!seconds) return "0:00";
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Handle download report
  const handleDownloadReport = () => {
    setIsDownloading(true);
    try {
      // Create a well-formatted report in JSON
      const reportData = {
        id: analysis.id,
        fileName: analysis.fileName,
        analysisDate: new Date(analysis.uploadDate).toLocaleString(),
        result: analysis.analysis.isDeepfake ? "Deepfake Detected" : "Authentic Video",
        confidenceScore: analysis.analysis.confidence + "%",
        processingTime: analysis.analysis.processingTime + " seconds",
        issues: analysis.analysis.issues || [],
        findings: analysis.analysis.findings || [],
        analysisDetails: analysis.analysis
      };
      
      // Create the blob and trigger download
      const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `analysis-report-${analysis.fileName.replace(/\.[^/.]+$/, "")}.json`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      toast({
        title: "Report Downloaded",
        description: `Analysis report for ${analysis.fileName} has been saved to your device.`
      });
    } catch (error) {
      toast({
        title: "Download Failed",
        description: "There was a problem creating your report. Please try again.",
        variant: "destructive"
      });
      console.error("Download error:", error);
    } finally {
      setIsDownloading(false);
    }
  };

  // Handle share
  const handleShare = async () => {
    setIsSharing(true);
    try {
      // Check if Web Share API is available
      if (navigator.share) {
        await navigator.share({
          title: `DeepGuard Analysis: ${analysis.fileName}`,
          text: `Video analysis report for ${analysis.fileName}. Confidence: ${analysis.analysis.confidence}%`,
          url: window.location.href
        });
        
        toast({
          title: "Shared Successfully",
          description: "The analysis has been shared."
        });
      } else {
        // Fallback to copying URL to clipboard
        await navigator.clipboard.writeText(window.location.href);
        
        toast({
          title: "Link Copied",
          description: "Analysis link copied to clipboard. You can now paste and share it."
        });
      }
    } catch (error) {
      // User cancelled or sharing failed
      if (error.name !== "AbortError") {
        toast({
          title: "Sharing Failed",
          description: "There was a problem sharing this analysis. Try copying the URL manually.",
          variant: "destructive"
        });
        console.error("Share error:", error);
      }
    } finally {
      setIsSharing(false);
    }
  };

  // Generate formatted date for display
  const formattedDate = new Date(analysis.uploadDate).toLocaleDateString("en-US", {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  // Calculate duration string to display
  const durationStr = analysis.analysis && analysis.analysis.processingTime ? 
    formatDuration(analysis.analysis.processingTime) : "0:00";

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
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
            <p className="text-muted-foreground">
              {analysis.fileName} 
              {analysis.fileSize ? ` • ${(analysis.fileSize / (1024 * 1024)).toFixed(1)} MB` : ''} 
              {durationStr !== "0:00" ? ` • ${durationStr} minutes` : ''} 
              {` • Analyzed ${formattedDate}`}
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <Button 
              variant="outline"
              className="rounded-lg glass-dark"
              onClick={handleShare} 
              disabled={isSharing}
            >
              {isSharing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Sharing...
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><path d="m16 6-4-4-4 4"/><path d="M12 2v13"/></svg>
                  Share
                </>
              )}
            </Button>
            <Button
              className="py-2 px-4 rounded-lg bg-primary text-black font-medium"
              onClick={handleDownloadReport}
              disabled={isDownloading}
            >
              {isDownloading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Downloading...
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                  Download Report
                </>
              )}
            </Button>
          </div>
        </div>
        
        {analysis.analysis ? (
          <>
            <AnalysisSummary analysis={analysis.analysis} />
            <VideoAnalysis analysis={analysis.analysis} fileName={analysis.fileName} />
            <DeepfakeDetails findings={(analysis.analysis && analysis.analysis.findings) || []} />
          </>
        ) : (
          <div className="glass rounded-xl p-8 text-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-4 text-muted-foreground">
              <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"/>
              <path d="M12 16v-4"/>
              <path d="M12 8h.01"/>
            </svg>
            <h2 className="text-2xl font-bold mb-2">Analysis Not Available</h2>
            <p className="text-muted-foreground mb-6">
              The analysis for this video file is not available or still processing.
              Please check back later or try uploading the file again.
            </p>
            <Button onClick={() => navigate("/dashboard")}>Return to Dashboard</Button>
          </div>
        )}
      </div>
    </div>
  );
}
