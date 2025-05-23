import React from "react";

export default function ParticlesBackground() {
  // Create a simpler background that won't cause errors
  return (
    <div 
      className="absolute top-0 left-0 w-full h-full z-[-1] overflow-hidden"
      style={{ 
        background: "linear-gradient(135deg, rgba(0,30,60,0.6) 0%, rgba(0,10,30,0.8) 100%)"
      }}
    >
      {/* Static dots for decoration */}
      {Array.from({ length: 30 }).map((_, i) => {
        // Pre-calculate these to avoid React warning about hooks
        const dotSize = 2 + Math.floor(i % 4);
        const topPos = `${Math.floor(i * 3.33) % 100}%`;
        const leftPos = `${Math.floor(i * 7.77) % 100}%`;
        const opacityVal = 0.2 + ((i % 5) / 10);
        
        return (
          <div
            key={i}
            className="absolute rounded-full"
            style={{
              backgroundColor: i % 3 === 0 ? "#00ff88" : i % 3 === 1 ? "#7000ff" : "#0088ff",
              width: `${dotSize}px`,
              height: `${dotSize}px`,
              top: topPos,
              left: leftPos,
              opacity: opacityVal,
              transform: `translate(${(i % 7) * 10}px, ${(i % 5) * 10}px)`,
            }}
          />
        );
      })}
    </div>
  );
}
