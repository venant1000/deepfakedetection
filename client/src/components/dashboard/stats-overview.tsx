import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useQuery } from "@tanstack/react-query";
import { VideoAnalysisResult } from "@shared/schema";
import { Loader2 } from "lucide-react";

export default function StatsOverview() {
  const { user } = useAuth();
  const [stats, setStats] = useState<any[]>([]);
  
  // Fetch real user video data from the database
  const { data: videoAnalyses, isLoading } = useQuery<VideoAnalysisResult[]>({
    queryKey: ["/api/videos"],
    enabled: !!user
  });
  
  // Calculate real stats based on the user's video analyses
  useEffect(() => {
    if (!videoAnalyses) {
      // Default empty stats while loading
      setStats([
        {
          title: "Videos Analyzed",
          value: "0",
          change: "No data yet",
          icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 8-6-6H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M10 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/><path d="m22 16-5.23-5.23a1 1 0 0 0-1.41 0L12 14.12l-1.36-1.36a1 1 0 0 0-1.41 0L2 20"/></svg>,
          bgColor: "bg-blue-500/20",
          textColor: "text-blue-400"
        },
        {
          title: "Deepfakes Detected",
          value: "0",
          change: "No data yet",
          icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>,
          bgColor: "bg-red-500/20",
          textColor: "text-red-400"
        },
        {
          title: "Storage Used",
          value: "0 MB",
          change: "10 GB remaining",
          icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
          bgColor: "bg-purple-500/20",
          textColor: "text-purple-400"
        },
        {
          title: "Accuracy Rate",
          value: "0%",
          change: "No data yet",
          icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>,
          bgColor: "bg-green-500/20",
          textColor: "text-green-400"
        }
      ]);
      return;
    }
    
    // Calculate real statistics from the video analyses
    const totalVideos = videoAnalyses.length;
    const deepfakes = videoAnalyses.filter(vid => vid.analysis.isDeepfake).length;
    
    // Calculate total storage used
    let totalStorage = 0;
    videoAnalyses.forEach(vid => {
      if (vid.fileSize) {
        totalStorage += vid.fileSize;
      }
    });
    
    // Calculate average accuracy (based on confidence values)
    let totalConfidence = 0;
    videoAnalyses.forEach(vid => {
      totalConfidence += vid.analysis.confidence;
    });
    const avgConfidence = totalVideos > 0 ? Math.round(totalConfidence / totalVideos) : 0;
    
    // Format storage value
    let storageValue;
    let storageRemaining;
    if (totalStorage < 1024) {
      storageValue = `${totalStorage} KB`;
      storageRemaining = "25 GB remaining";
    } else if (totalStorage < 1024 * 1024) {
      storageValue = `${(totalStorage / 1024).toFixed(1)} MB`;
      storageRemaining = "24.9 GB remaining";
    } else {
      storageValue = `${(totalStorage / (1024 * 1024)).toFixed(1)} GB`;
      const remaining = 25 - (totalStorage / (1024 * 1024));
      storageRemaining = `${remaining.toFixed(1)} GB remaining`;
    }
    
    // Update stats with real data
    setStats([
      {
        title: "Videos Analyzed",
        value: totalVideos.toString(),
        change: totalVideos > 0 ? "Recently uploaded" : "No videos yet",
        isPositive: totalVideos > 0,
        icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 8-6-6H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M10 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/><path d="m22 16-5.23-5.23a1 1 0 0 0-1.41 0L12 14.12l-1.36-1.36a1 1 0 0 0-1.41 0L2 20"/></svg>,
        bgColor: "bg-blue-500/20",
        textColor: "text-blue-400"
      },
      {
        title: "Deepfakes Detected",
        value: deepfakes.toString(),
        change: totalVideos > 0 ? `${Math.round((deepfakes / totalVideos) * 100)}% of uploads` : "No videos yet",
        isPositive: deepfakes === 0,
        icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>,
        bgColor: "bg-red-500/20",
        textColor: "text-red-400"
      },
      {
        title: "Storage Used",
        value: storageValue,
        change: storageRemaining,
        icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
        bgColor: "bg-purple-500/20",
        textColor: "text-purple-400"
      },
      {
        title: "Accuracy Rate",
        value: `${avgConfidence}%`,
        change: totalVideos > 0 ? "Based on confidence scores" : "No videos yet",
        isPositive: avgConfidence > 80,
        icon: <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>,
        bgColor: "bg-green-500/20",
        textColor: "text-green-400"
      }
    ]);
  }, [videoAnalyses]);
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {stats.map((stat, index) => (
        <div key={index} className="glass rounded-xl p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-muted-foreground text-sm">{stat.title}</p>
              <p className="text-3xl font-semibold">{stat.value}</p>
            </div>
            <div className={`h-10 w-10 rounded-full flex items-center justify-center ${stat.bgColor} ${stat.textColor}`}>
              {stat.icon}
            </div>
          </div>
          <div className="flex items-center text-sm">
            {'isPositive' in stat ? (
              <>
                <span className={stat.isPositive ? "text-primary mr-1" : "text-[#ff3366] mr-1"}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="inline">
                    {stat.isPositive ? (
                      <path d="m18 15-6-6-6 6"/>
                    ) : (
                      <path d="m6 9 6 6 6-6"/>
                    )}
                  </svg> {stat.change}
                </span>
                <span className="text-muted-foreground">from last month</span>
              </>
            ) : (
              <span className="text-muted-foreground">{stat.change}</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
