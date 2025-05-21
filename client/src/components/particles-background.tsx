import { useEffect, useRef } from "react";

export default function ParticlesBackground() {
  const particlesContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Only load particles if the container exists
    if (!particlesContainerRef.current) return;

    // Load tsParticles
    const loadParticles = async () => {
      try {
        // Dynamically import tsParticles to avoid SSR issues
        const tsParticles = (await import('tsparticles')).tsParticles;
        
        await tsParticles.load("particles-container", {
          fpsLimit: 60,
          particles: {
            number: {
              value: 50,
              density: {
                enable: true,
                value_area: 800
              }
            },
            color: {
              value: ["#00ff88", "#7000ff", "#0088ff"]
            },
            shape: {
              type: "circle"
            },
            opacity: {
              value: 0.5,
              random: true
            },
            size: {
              value: 3,
              random: true
            },
            move: {
              enable: true,
              speed: 1,
              direction: "none",
              random: true,
              straight: false,
              out_mode: "out",
              bounce: false
            }
          },
          interactivity: {
            detect_on: "canvas",
            events: {
              onhover: {
                enable: true,
                mode: "grab"
              },
              onclick: {
                enable: true,
                mode: "push"
              },
              resize: true
            },
            modes: {
              grab: {
                distance: 140,
                line_linked: {
                  opacity: 0.8
                }
              },
              push: {
                particles_nb: 3
              }
            }
          },
          retina_detect: true
        });
      } catch (error) {
        console.error("Failed to load particles:", error);
      }
    };

    loadParticles();

    // Cleanup function
    return () => {
      try {
        const tsParticles = (window as any).tsParticles;
        if (tsParticles && tsParticles.destroy) {
          tsParticles.destroy("particles-container");
        }
      } catch (error) {
        console.error("Failed to destroy particles:", error);
      }
    };
  }, []);

  return (
    <div
      id="particles-container"
      ref={particlesContainerRef}
      className="absolute top-0 left-0 w-full h-full z-[-1]"
    />
  );
}
