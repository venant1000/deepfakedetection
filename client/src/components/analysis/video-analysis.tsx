import { useEffect, useState, useRef } from "react";

interface TimelineMarker {
  position: number;
  tooltip: string;
  type: "normal" | "warning" | "danger";
}

interface VideoAnalysisProps {
  analysis: {
    timeline: TimelineMarker[];
  };
}

export default function VideoAnalysis({ analysis }: VideoAnalysisProps) {
  const [progressPosition, setProgressPosition] = useState(35);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState("1:18");
  const [isPlaying, setIsPlaying] = useState(false);

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
          const totalSeconds = 222; // 3:42 in seconds
          const currentSeconds = Math.floor(totalSeconds * (newValue / 100));
          const minutes = Math.floor(currentSeconds / 60);
          const seconds = currentSeconds % 60;
          setCurrentTime(`${minutes}:${seconds.toString().padStart(2, '0')}`);
          
          return newValue;
        });
      }, 100);
      
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

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
    const totalSeconds = 222; // 3:42 in seconds
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
          <div className="aspect-video rounded-lg bg-black flex items-center justify-center relative">
            {/* Video placeholder */}
            <div className="text-muted-foreground">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="48" 
                height="48" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                className="mb-2 mx-auto"
              >
                <path d="m22 8-6-6H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <path d="M14 2v6h6"/>
                <path d="M10 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/>
                <path d="m22 16-5.23-5.23a1 1 0 0 0-1.41 0L12 14.12l-1.36-1.36a1 1 0 0 0-1.41 0L2 20"/>
              </svg>
              <span>Video Player</span>
            </div>
            
            {/* Playback controls */}
            <div className="absolute bottom-0 left-0 right-0 p-4 glass-dark">
              <div className="flex items-center gap-3">
                <button className="text-white" onClick={handlePlayPause}>
                  {isPlaying ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                  )}
                </button>
                
                <div 
                  className="h-1 bg-muted flex-grow rounded-full relative cursor-pointer" 
                  ref={timelineRef}
                  onClick={handleTimelineClick}
                >
                  <div 
                    className="h-1 bg-primary rounded-full absolute top-0 left-0" 
                    style={{ width: `${progressPosition}%` }}
                  ></div>
                </div>
                
                <span className="text-sm text-white">{currentTime} / 3:42</span>
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
            
            {/* AI analysis overlay elements */}
            <div className="absolute top-4 left-4 glass p-2 rounded-md text-xs font-mono">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-[#ff3366] animate-pulse"></div>
                <span>LIP-SYNC ERROR DETECTED</span>
              </div>
            </div>
            
            <div className="absolute bottom-16 right-4 glass p-2 rounded-md text-xs font-mono">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-[#ffbb00] animate-pulse"></div>
                <span>UNNATURAL EYE MOVEMENT</span>
              </div>
            </div>
            
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 glass p-4 rounded-lg text-center">
              <div className="text-lg font-semibold mb-2">Frame Analysis</div>
              <div className="text-sm text-muted-foreground mb-4">Multiple facial inconsistencies detected</div>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="glass-dark p-2 rounded">
                  <div className="font-semibold text-[#ff3366] mb-1">Lip Movement</div>
                  <div>94% anomaly</div>
                </div>
                <div className="glass-dark p-2 rounded">
                  <div className="font-semibold text-[#ffbb00] mb-1">Eye Blinking</div>
                  <div>78% anomaly</div>
                </div>
                <div className="glass-dark p-2 rounded">
                  <div className="font-semibold text-[#ffbb00] mb-1">Facial Texture</div>
                  <div>71% anomaly</div>
                </div>
                <div className="glass-dark p-2 rounded">
                  <div className="font-semibold text-primary mb-1">Head Position</div>
                  <div>22% anomaly</div>
                </div>
              </div>
            </div>
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
            <span>0:45</span>
            <span>1:30</span>
            <span>2:15</span>
            <span>3:00</span>
            <span>3:42</span>
          </div>
        </div>
      </div>
    </div>
  );
}
