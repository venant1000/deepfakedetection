import { useEffect, useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Loader2, AlertCircle, Info } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface TimelineMarker {
  position: number;
  tooltip: string;
  type: "normal" | "warning" | "danger";
  analysis?: string; // Gemini-powered detailed analysis
}

interface VideoAnalysisProps {
  analysis: {
    timeline: TimelineMarker[];
    videoUrl?: string;
  };
}

export default function VideoAnalysis({ analysis }: VideoAnalysisProps) {
  const { toast } = useToast();
  const [progressPosition, setProgressPosition] = useState(0);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState("0:00");
  const [isPlaying, setIsPlaying] = useState(false);
  const [totalDuration, setTotalDuration] = useState("2:30");
  const [totalSeconds, setTotalSeconds] = useState(150); // Default 2:30 in seconds
  const [analyzingMarkers, setAnalyzingMarkers] = useState(false);
  const [selectedMarker, setSelectedMarker] = useState<TimelineMarker | null>(null);
  const [enrichedTimeline, setEnrichedTimeline] = useState<TimelineMarker[]>([]);
  const [loadingMarkerDetails, setLoadingMarkerDetails] = useState<Record<number, boolean>>({});

  // Set initial duration when the component loads
  useEffect(() => {
    // If we had actual video duration, we would parse it from analysis data
    // For now we'll use the default until real duration metadata is available
    setTotalDuration("2:30"); // Default duration
    setTotalSeconds(150); // 2:30 in seconds
    
    // Initialize the enriched timeline with the analysis data
    if (analysis && analysis.timeline) {
      setEnrichedTimeline([...analysis.timeline]);
    }
  }, [analysis]);
  
  // Function to fetch AI analysis for a specific timeline marker
  const getMarkerAnalysis = async (marker: TimelineMarker, index: number) => {
    try {
      setLoadingMarkerDetails(prev => ({ ...prev, [index]: true }));
      
      // Format timestamp for the marker
      const totalSecondsAtPosition = Math.floor((marker.position/100) * totalSeconds);
      const minutes = Math.floor(totalSecondsAtPosition / 60);
      const seconds = totalSecondsAtPosition % 60;
      const timestamp = `${minutes}:${seconds.toString().padStart(2, '0')}`;
      
      // Make API request to get the detailed analysis
      const response = await fetch('/api/analyze-timeline-marker', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          markerType: marker.type,
          markerTooltip: marker.tooltip,
          timestamp
        }),
        credentials: 'include'
      });
      
      // Parse response data with proper typing
      const data: { analysis: string } = await response.json();
      
      if (response.ok && data && data.analysis) {
        // Update the enriched timeline with the analysis
        setEnrichedTimeline(prev => {
          const updated = [...prev];
          updated[index] = {
            ...prev[index],
            analysis: data.analysis
          };
          return updated;
        });
        
        // Update selected marker if it's the one we were looking at
        if (selectedMarker && selectedMarker.position === marker.position) {
          setSelectedMarker({
            ...selectedMarker,
            analysis: data.analysis
          });
        }
        
        return data.analysis;
      } else {
        throw new Error("Failed to get analysis from the API");
      }
    } catch (error) {
      console.error('Error fetching marker analysis:', error);
      toast({
        title: "Analysis failed",
        description: "We couldn't get the detailed explanation for this anomaly.",
        variant: "destructive"
      });
    } finally {
      setLoadingMarkerDetails(prev => ({ ...prev, [index]: false }));
    }
  };
  
  // Function to analyze all timeline markers at once
  const analyzeAllMarkers = async () => {
    try {
      setAnalyzingMarkers(true);
      
      const promiseArray = analysis.timeline.map((marker, index) => getMarkerAnalysis(marker, index));
      await Promise.all(promiseArray);
      
      toast({
        title: "Analysis complete",
        description: "All timeline markers have been analyzed with AI."
      });
    } catch (error) {
      console.error('Error analyzing all markers:', error);
      toast({
        title: "Analysis failed",
        description: "There was an error analyzing some of the timeline markers.",
        variant: "destructive"
      });
    } finally {
      setAnalyzingMarkers(false);
    }
  };

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setProgressPosition((prev) => {
          const newValue = prev + 0.2;
          if (newValue >= 100) {
            setIsPlaying(false);
            return 100;
          }
          
          // Update current time based on progress
          const currentSeconds = Math.floor(totalSeconds * (newValue / 100));
          const minutes = Math.floor(currentSeconds / 60);
          const seconds = currentSeconds % 60;
          setCurrentTime(`${minutes}:${seconds.toString().padStart(2, '0')}`);
          
          return newValue;
        });
      }, 100);
      
      return () => clearInterval(interval);
    }
  }, [isPlaying, totalSeconds]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current) return;
    
    const rect = timelineRef.current.getBoundingClientRect();
    const clickPosition = e.clientX - rect.left;
    const newPosition = (clickPosition / rect.width) * 100;
    
    setProgressPosition(Math.min(Math.max(newPosition, 0), 100));
    
    // Update current time based on new position
    const currentSeconds = Math.floor(totalSeconds * (newPosition / 100));
    const minutes = Math.floor(currentSeconds / 60);
    const seconds = currentSeconds % 60;
    setCurrentTime(`${minutes}:${seconds.toString().padStart(2, '0')}`);
  };

  return (
    <div className="glass rounded-xl p-6 mb-8">
      <h2 className="text-xl font-semibold mb-6">Video Analysis</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Original Video */}
        <div>
          <h3 className="text-lg font-medium mb-3">Original Video</h3>
          <div className="aspect-video rounded-lg bg-black flex items-center justify-center relative overflow-hidden">
            {/* Actual video player */}
            <video 
              className="w-full h-full object-contain"
              src={analysis.videoUrl}
              ref={(video) => {
                if (video) {
                  video.addEventListener('loadedmetadata', () => {
                    setTotalSeconds(Math.floor(video.duration));
                    setTotalDuration(`${Math.floor(video.duration / 60)}:${String(Math.floor(video.duration % 60)).padStart(2, '0')}`);
                  });
                  
                  video.addEventListener('timeupdate', () => {
                    const percent = (video.currentTime / video.duration) * 100;
                    setProgressPosition(percent);
                    
                    const minutes = Math.floor(video.currentTime / 60);
                    const seconds = Math.floor(video.currentTime % 60);
                    setCurrentTime(`${minutes}:${String(seconds).padStart(2, '0')}`);
                  });
                  
                  // Update playing state when video ends
                  video.addEventListener('ended', () => {
                    setIsPlaying(false);
                  });
                }
              }}
              onClick={handlePlayPause}
            />
            
            {/* Playback controls */}
            <div className="absolute bottom-0 left-0 right-0 p-4 glass-dark">
              <div className="flex items-center gap-3">
                <button className="text-white" onClick={(e) => {
                  e.stopPropagation();
                  handlePlayPause();
                  
                  // Play/pause the actual video
                  const parentDiv = e.currentTarget.parentElement;
                  const videoContainer = parentDiv ? parentDiv.parentElement : null;
                  const videoElement = videoContainer ? videoContainer.querySelector('video') : null;
                  
                  if (videoElement) {
                    if (isPlaying) {
                      videoElement.pause();
                    } else {
                      videoElement.play().catch(e => console.error('Error playing video:', e));
                    }
                  }
                }}>
                  {isPlaying ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                  )}
                </button>
                
                <div 
                  className="h-1 bg-muted flex-grow rounded-full relative cursor-pointer" 
                  ref={timelineRef}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleTimelineClick(e);
                    
                    // Get video element using a more reliable approach
                    const parentDiv = e.currentTarget.parentElement;
                    const videoContainer = parentDiv ? parentDiv.parentElement : null;
                    const videoElement = videoContainer ? videoContainer.querySelector('video') : null;
                    
                    if (videoElement && timelineRef.current) {
                      const rect = timelineRef.current.getBoundingClientRect();
                      const clickPosition = e.clientX - rect.left;
                      const newPosition = (clickPosition / rect.width);
                      videoElement.currentTime = newPosition * videoElement.duration;
                    }
                  }}
                >
                  <div 
                    className="h-1 bg-primary rounded-full absolute top-0 left-0" 
                    style={{ width: `${progressPosition}%` }}
                  ></div>
                </div>
                
                <span className="text-sm text-white">{currentTime} / {totalDuration}</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* AI Analysis View */}
        <div>
          <h3 className="text-lg font-medium mb-3">AI Analysis View</h3>
          <div className="aspect-video rounded-lg glass-dark flex items-center justify-center relative overflow-hidden">
            {/* Analysis visualization */}
            <div className="absolute inset-0 bg-gradient-to-br from-black/50 to-black/70 flex items-center justify-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="75%"
                height="75%"
                viewBox="0 0 24 24"
                fill="none"
                stroke="rgba(0, 255, 136, 0.15)"
                strokeWidth="0.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="absolute"
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M8 12h8" />
                <path d="M12 8v8" />
              </svg>
            </div>
            
            {/* Display issues from analysis data */}
            {analysis.timeline && analysis.timeline.length > 0 ? (
              <>
                {/* Show first issue at top */}
                {analysis.timeline[0] && (
                  <div className="absolute top-4 left-4 glass p-2 rounded-md text-xs font-mono">
                    <div className="flex items-center gap-2">
                      <div className={`h-2 w-2 rounded-full ${
                        analysis.timeline[0].type === 'danger' ? 'bg-[#ff3366]' : 
                        analysis.timeline[0].type === 'warning' ? 'bg-[#ffbb00]' : 
                        'bg-primary'
                      } animate-pulse`}></div>
                      <span>{analysis.timeline[0].tooltip.toUpperCase()}</span>
                    </div>
                  </div>
                )}
                
                {/* Show second issue at bottom */}
                {analysis.timeline[1] && (
                  <div className="absolute bottom-16 right-4 glass p-2 rounded-md text-xs font-mono">
                    <div className="flex items-center gap-2">
                      <div className={`h-2 w-2 rounded-full ${
                        analysis.timeline[1].type === 'danger' ? 'bg-[#ff3366]' : 
                        analysis.timeline[1].type === 'warning' ? 'bg-[#ffbb00]' : 
                        'bg-primary'
                      } animate-pulse`}></div>
                      <span>{analysis.timeline[1].tooltip.toUpperCase()}</span>
                    </div>
                  </div>
                )}
                
                {/* Central analysis summary - generated from the timeline */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 glass p-4 rounded-lg text-center">
                  <div className="text-lg font-semibold mb-2">Frame Analysis</div>
                  <div className="text-sm text-muted-foreground mb-4">
                    {analysis.timeline.filter(m => m.type === 'danger').length > 0 ? 
                      "Multiple critical issues detected" : 
                      analysis.timeline.filter(m => m.type === 'warning').length > 0 ?
                      "Suspicious elements identified" : 
                      "No significant issues detected"}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    {analysis.timeline.slice(0, 4).map((marker, idx) => (
                      <div key={idx} className="glass-dark p-2 rounded">
                        <div className={`font-semibold ${
                          marker.type === 'danger' ? 'text-[#ff3366]' : 
                          marker.type === 'warning' ? 'text-[#ffbb00]' : 
                          'text-primary'
                        } mb-1`}>
                          {marker.tooltip.split(' ')[0]}
                        </div>
                        <div>
                          {marker.type === 'danger' ? '85-95% anomaly' : 
                           marker.type === 'warning' ? '65-85% anomaly' : 
                           '20-40% anomaly'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center">
                <div className="text-lg font-semibold">No Anomalies Detected</div>
                <div className="text-sm text-muted-foreground mt-2">Video appears authentic</div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Timeline */}
      <div className="mt-8">
        <h3 className="text-lg font-medium mb-4">Anomaly Timeline</h3>
        <div className="relative">
          {/* Timeline bar */}
          <div 
            className="h-1 bg-muted rounded-full mb-6 relative cursor-pointer"
            ref={timelineRef}
            onClick={handleTimelineClick}
          >
            {/* Progress indicator */}
            <div 
              className="h-1 bg-muted-foreground rounded-full absolute top-0 left-0" 
              style={{ width: `${progressPosition}%` }}
            ></div>
            
            {/* Timeline markers */}
            {analysis.timeline.map((marker, index) => (
              <div 
                key={index}
                className={`timeline-marker ${marker.type === 'warning' ? 'warning' : marker.type === 'danger' ? 'danger' : ''}`}
                style={{ left: `${marker.position}%` }}
              >
                <div className="timeline-tooltip">{marker.tooltip}</div>
              </div>
            ))}
          </div>
          
          {/* Time markers */}
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>0:00</span>
            <span>0:30</span>
            <span>1:00</span>
            <span>1:30</span>
            <span>2:00</span>
            <span>{totalDuration}</span>
          </div>
          
          {/* Timeline explanations - Grid Layout */}
          <div className="mt-6 space-y-4">
            <div className="flex justify-between items-center">
              <h4 className="font-medium">Detection Markers Explained:</h4>
              {enrichedTimeline.length > 0 && (
                <Button 
                  size="sm" 
                  onClick={analyzeAllMarkers} 
                  disabled={analyzingMarkers}
                  className="bg-primary/90 hover:bg-primary"
                >
                  {analyzingMarkers ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Info className="mr-2 h-4 w-4" />
                      AI Analyze All Markers
                    </>
                  )}
                </Button>
              )}
            </div>
            
            {/* Add explanation about the AI-powered analysis */}
            <div className="flex items-center p-3 rounded-lg bg-secondary/10 text-sm">
              <AlertCircle className="h-5 w-5 mr-2 text-secondary" />
              <p>
                Click on any marker to see a detailed AI-powered analysis based on the severity level. 
                Markers with different colors indicate different severity levels of potential manipulations.
              </p>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-3">
              {enrichedTimeline.map((marker, index) => {
                // Calculate timestamp for display
                const markerTimestamp = `${Math.floor((marker.position/100) * totalSeconds / 60)}:${String(Math.floor((marker.position/100) * totalSeconds % 60)).padStart(2, '0')}`;
                
                return (
                  <div 
                    key={index} 
                    className={`glass-dark p-4 rounded-lg border-l-4 transition-all hover:shadow-md cursor-pointer ${
                      marker.type === 'danger' ? 'border-[#ff3366]' : 
                      marker.type === 'warning' ? 'border-[#ffbb00]' : 
                      'border-primary'
                    } ${selectedMarker === marker ? 'ring-2 ring-primary/50' : ''}`}
                    onClick={() => {
                      if (marker.analysis) {
                        setSelectedMarker(marker);
                      } else {
                        getMarkerAnalysis(marker, index);
                        setSelectedMarker(marker);
                      }
                    }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className={`h-3 w-3 rounded-full ${
                          marker.type === 'danger' ? 'bg-[#ff3366]' : 
                          marker.type === 'warning' ? 'bg-[#ffbb00]' : 
                          'bg-primary'
                        }`}></div>
                        <div className="font-medium text-sm">
                          {marker.type === 'danger' ? 'High Risk' : 
                           marker.type === 'warning' ? 'Medium Risk' : 
                           'Low Risk'}
                        </div>
                      </div>
                      
                      {!marker.analysis && !loadingMarkerDetails[index] && (
                        <Button 
                          size="sm" 
                          variant="ghost" 
                          className="h-7 px-2"
                          onClick={(e) => {
                            e.stopPropagation();
                            getMarkerAnalysis(marker, index);
                          }}
                        >
                          <Info className="h-4 w-4" />
                        </Button>
                      )}
                      
                      {loadingMarkerDetails[index] && (
                        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                      )}
                    </div>
                    
                    <div className="text-sm font-medium mb-1">
                      {marker.tooltip}
                    </div>
                    
                    <div className="text-xs text-muted-foreground mb-2">
                      Timestamp: {markerTimestamp}
                    </div>
                    
                    {marker.analysis ? (
                      <div className="text-xs bg-black/20 p-3 rounded my-2 border-l-2 border-primary/50">
                        <p className="italic text-primary-foreground">{marker.analysis}</p>
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        {marker.type === 'danger' 
                          ? 'Critical manipulation detected with high confidence.' 
                          : marker.type === 'warning' 
                          ? 'Potential inconsistency that suggests manipulation.' 
                          : 'Normal variation within expected parameters.'}
                      </p>
                    )}
                    
                    {!marker.analysis && (
                      <Button
                        variant="link" 
                        className="text-xs p-0 h-auto mt-2 text-primary hover:text-primary/80"
                        onClick={(e) => {
                          e.stopPropagation();
                          getMarkerAnalysis(marker, index);
                        }}
                        disabled={loadingMarkerDetails[index]}
                      >
                        {loadingMarkerDetails[index] ? (
                          <>
                            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          "Get AI Analysis"
                        )}
                      </Button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
