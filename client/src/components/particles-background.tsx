import { useCallback } from "react";
import Particles from "@tsparticles/react";
import { Engine } from "@tsparticles/engine";
import { loadSlim } from "@tsparticles/slim";

export default function ParticlesBackground() {
  const particlesInit = useCallback(async (engine: Engine) => {
    // Initialize the tsParticles instance
    await loadSlim(engine);
  }, []);

  return (
    <Particles
      id="particles-background"
      className="absolute top-0 left-0 w-full h-full z-[-1]"
      init={particlesInit}
      options={{
        background: {
          color: {
            value: "transparent",
          },
        },
        fpsLimit: 60,
        particles: {
          color: {
            value: ["#00ff88", "#7000ff", "#0088ff"]
          },
          number: {
            value: 50,
            density: {
              enable: true,
              value_area: 800
            }
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
            outModes: "out",
            bounce: false
          }
        },
        interactivity: {
          detectsOn: "canvas",
          events: {
            onHover: {
              enable: true,
              mode: "grab"
            },
            onClick: {
              enable: true,
              mode: "push"
            },
            resize: true
          },
          modes: {
            grab: {
              distance: 140,
              links: {
                opacity: 0.8
              }
            },
            push: {
              quantity: 3
            }
          }
        },
        detectRetina: true
      }}
    />
  );
}
