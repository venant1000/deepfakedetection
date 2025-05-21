import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

export default function AnalyticsDashboard() {
  const usageChartRef = useRef<HTMLCanvasElement | null>(null);
  const detectionChartRef = useRef<HTMLCanvasElement | null>(null);
  const performanceChartRef = useRef<HTMLCanvasElement | null>(null);
  
  const usageChartInstance = useRef<Chart | null>(null);
  const detectionChartInstance = useRef<Chart | null>(null);
  const performanceChartInstance = useRef<Chart | null>(null);
  
  useEffect(() => {
    // Initialize charts when the component mounts
    initializeCharts();

    // Cleanup charts when the component unmounts
    return () => {
      if (usageChartInstance.current) {
        usageChartInstance.current.destroy();
      }
      if (detectionChartInstance.current) {
        detectionChartInstance.current.destroy();
      }
      if (performanceChartInstance.current) {
        performanceChartInstance.current.destroy();
      }
    };
  }, []);

  const initializeCharts = () => {
    if (usageChartRef.current) {
      const ctx = usageChartRef.current.getContext('2d');
      if (ctx) {
        usageChartInstance.current = new Chart(ctx, {
          type: 'line',
          data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
              label: 'Videos Analyzed',
              data: [5300, 8600, 12400, 15700, 19200, 23500],
              borderColor: 'hsl(var(--primary))',
              backgroundColor: 'hsla(var(--primary), 0.1)',
              tension: 0.3,
              fill: true
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                display: false
              },
              tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(10, 10, 10, 0.8)',
                titleColor: 'white',
                bodyColor: 'white',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1
              }
            },
            scales: {
              y: {
                beginAtZero: true,
                grid: {
                  color: 'rgba(255, 255, 255, 0.05)'
                },
                ticks: {
                  color: 'rgba(255, 255, 255, 0.7)'
                }
              },
              x: {
                grid: {
                  color: 'rgba(255, 255, 255, 0.05)'
                },
                ticks: {
                  color: 'rgba(255, 255, 255, 0.7)'
                }
              }
            }
          }
        });
      }
    }

    if (detectionChartRef.current) {
      const ctx = detectionChartRef.current.getContext('2d');
      if (ctx) {
        detectionChartInstance.current = new Chart(ctx, {
          type: 'doughnut',
          data: {
            labels: ['Authentic', 'Deepfake', 'Suspicious'],
            datasets: [{
              data: [45, 35, 20],
              backgroundColor: [
                'hsl(var(--primary))',
                'hsl(350, 100%, 60%)',
                'hsl(45, 100%, 50%)'
              ],
              borderWidth: 0
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'bottom',
                labels: {
                  color: 'rgba(255, 255, 255, 0.7)',
                  padding: 20,
                  font: {
                    size: 12
                  }
                }
              },
              tooltip: {
                backgroundColor: 'rgba(10, 10, 10, 0.8)',
                titleColor: 'white',
                bodyColor: 'white',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1
              }
            },
            cutout: '70%'
          }
        });
      }
    }

    if (performanceChartRef.current) {
      const ctx = performanceChartRef.current.getContext('2d');
      if (ctx) {
        performanceChartInstance.current = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: ['API Latency', 'Video Processing', 'Authentication', 'Database Queries', 'Report Generation'],
            datasets: [{
              label: 'Performance (ms)',
              data: [42, 356, 18, 25, 85],
              backgroundColor: 'hsl(var(--secondary))'
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                display: false
              },
              tooltip: {
                backgroundColor: 'rgba(10, 10, 10, 0.8)',
                titleColor: 'white',
                bodyColor: 'white',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1
              }
            },
            scales: {
              y: {
                beginAtZero: true,
                grid: {
                  color: 'rgba(255, 255, 255, 0.05)'
                },
                ticks: {
                  color: 'rgba(255, 255, 255, 0.7)'
                }
              },
              x: {
                grid: {
                  display: false
                },
                ticks: {
                  color: 'rgba(255, 255, 255, 0.7)'
                }
              }
            }
          }
        });
      }
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      {/* Usage Overview */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">Usage Overview</h2>
        <div className="h-64">
          <canvas ref={usageChartRef}></canvas>
        </div>
      </div>
      
      {/* Deepfake Ratio */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">Detection Results</h2>
        <div className="h-64 flex items-center justify-center">
          <canvas ref={detectionChartRef}></canvas>
        </div>
      </div>
      
      {/* System Performance */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">System Performance</h2>
        <div className="h-64">
          <canvas ref={performanceChartRef}></canvas>
        </div>
      </div>
      
      {/* Geographic Distribution */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">Geographic Distribution</h2>
        <div className="h-64 flex items-center justify-center">
          {/* World map visualization placeholder */}
          <div className="text-center text-muted-foreground">
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              width="64" 
              height="64" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="1" 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              className="mb-4 mx-auto opacity-50"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              <path d="M2 12h20" />
              <path d="M12 2v20" />
            </svg>
            <p>Interactive world map showing usage hotspots</p>
          </div>
        </div>
      </div>
    </div>
  );
}
