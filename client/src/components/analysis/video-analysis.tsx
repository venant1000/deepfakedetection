import { useEffect, useState, useRef } from "react";

interface TimelineMarker {
  position: number;
  tooltip: string;
  type: "normal" | "warning" | "danger";
}

interface VideoAnalysisProps {
  analysis: {
    timeline: TimelineMarker[];
    videoUrl?: string;
  };
}

export default function VideoAnalysis({ analysis }: VideoAnalysisProps) {
  const [progressPosition, setProgressPosition] = useState(0);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState("0:00");
  const [isPlaying, setIsPlaying] = useState(false);
  const [totalDuration, setTotalDuration] = useState("2:30");
  const [totalSeconds, setTotalSeconds] = useState(150); // Default 2:30 in seconds

  // Set initial duration when the component loads
  useEffect(() => {
    // If we had actual video duration, we would parse it from analysis data
    // For now we'll use the default until real duration metadata is available
    setTotalDuration("2:30"); // Default duration
    setTotalSeconds(150); // 2:30 in seconds
  }, []);

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
              poster="/public/placeholder.svg"
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
          
          {/* Timeline explanations */}
          <div className="mt-6 space-y-4">
            <h4 className="font-medium text-sm">Detection Markers Explained:</h4>
            <ul className="space-y-3">
              {analysis.timeline.map((marker, index) => (
                <li key={index} className="flex items-start gap-3">
                  <div className={`mt-1 h-3 w-3 rounded-full ${
                    marker.type === 'danger' ? 'bg-[#ff3366]' : 
                    marker.type === 'warning' ? 'bg-[#ffbb00]' : 
                    'bg-primary'
                  } flex-shrink-0`}></div>
                  <div>
                    <div className="font-medium text-sm">
                      {marker.tooltip} 
                      <span className="text-muted-foreground text-xs ml-2">
                        at {Math.floor((marker.position/100) * totalSeconds / 60)}:{String(Math.floor((marker.position/100) * totalSeconds % 60)).padStart(2, '0')}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {marker.type === 'danger' 
                        ? 'Critical manipulation detected with high confidence.' 
                        : marker.type === 'warning' 
                        ? 'Potential inconsistency that suggests manipulation.' 
                        : 'Normal variation within expected parameters.'}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
