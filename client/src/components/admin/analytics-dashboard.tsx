import { useEffect, useRef, useState } from "react";
import Chart from "chart.js/auto";
import { Loader2 } from "lucide-react";

export default function AnalyticsDashboard() {
  const usageChartRef = useRef<HTMLCanvasElement | null>(null);
  const detectionChartRef = useRef<HTMLCanvasElement | null>(null);
  const performanceChartRef = useRef<HTMLCanvasElement | null>(null);
  const trendsChartRef = useRef<HTMLCanvasElement | null>(null);
  
  const usageChartInstance = useRef<Chart | null>(null);
  const detectionChartInstance = useRef<Chart | null>(null);
  const performanceChartInstance = useRef<Chart | null>(null);
  const trendsChartInstance = useRef<Chart | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [activeMetric, setActiveMetric] = useState("total");
  
  useEffect(() => {
    // Simulate loading delay
    const timer = setTimeout(() => {
      setIsLoading(false);
      // Initialize charts when the component mounts
      initializeCharts();
    }, 500);

    // Cleanup charts when the component unmounts
    return () => {
      clearTimeout(timer);
      if (usageChartInstance.current) {
        usageChartInstance.current.destroy();
      }
      if (detectionChartInstance.current) {
        detectionChartInstance.current.destroy();
      }
      if (performanceChartInstance.current) {
        performanceChartInstance.current.destroy();
      }
      if (trendsChartInstance.current) {
        trendsChartInstance.current.destroy();
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
            }, {
              label: 'New Users',
              data: [1200, 1900, 3100, 4200, 4800, 5600],
              borderColor: 'hsl(var(--secondary))',
              backgroundColor: 'transparent',
              tension: 0.3,
              borderDash: [5, 5]
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
              mode: 'index',
              intersect: false,
            },
            plugins: {
              legend: {
                position: 'top',
                labels: {
                  usePointStyle: true,
                  padding: 20,
                  color: 'rgba(255, 255, 255, 0.7)',
                  font: {
                    size: 12
                  }
                }
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
              borderWidth: 2,
              borderColor: 'rgba(10, 10, 10, 0.05)'
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
                  },
                  usePointStyle: true,
                  pointStyle: 'circle'
                }
              },
              tooltip: {
                backgroundColor: 'rgba(10, 10, 10, 0.8)',
                titleColor: 'white',
                bodyColor: 'white',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1,
                callbacks: {
                  label: function(context) {
                    const label = context.label || '';
                    const value = context.raw || 0;
                    const percentage = Math.round((value as number / 100) * 100);
                    return `${label}: ${percentage}%`;
                  }
                }
              }
            },
            cutout: '70%',
            animation: {
              animateScale: true,
              animateRotate: true
            }
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
              backgroundColor: [
                'hsla(var(--primary), 0.7)',
                'hsla(var(--primary), 0.8)',
                'hsla(var(--primary), 0.9)',
                'hsla(var(--primary), 1)',
                'hsla(var(--secondary), 0.8)'
              ],
              borderRadius: 6
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
                borderWidth: 1,
                callbacks: {
                  label: function(context) {
                    return `${context.raw}ms`;
                  }
                }
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
            },
            animation: {
              delay: (context) => context.dataIndex * 100
            }
          }
        });
      }
    }

    if (trendsChartRef.current) {
      const ctx = trendsChartRef.current.getContext('2d');
      if (ctx) {
        trendsChartInstance.current = new Chart(ctx, {
          type: 'line',
          data: {
            labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
            datasets: [{
              label: 'Detection Accuracy',
              data: [92, 94, 91, 95, 97, 96],
              borderColor: 'hsl(var(--primary))',
              backgroundColor: 'transparent',
              tension: 0.4,
              borderWidth: 3,
              pointBackgroundColor: 'hsl(var(--primary))',
              pointRadius: 4,
              pointHoverRadius: 6
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
                callbacks: {
                  label: function(context) {
                    return `Accuracy: ${context.raw}%`;
                  }
                }
              }
            },
            scales: {
              y: {
                beginAtZero: false,
                min: 85,
                max: 100,
                grid: {
                  color: 'rgba(255, 255, 255, 0.05)'
                },
                ticks: {
                  color: 'rgba(255, 255, 255, 0.7)',
                  callback: function(value) {
                    return value + '%';
                  }
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
  };

  const metricStats = {
    total: { value: 23500, change: "+12.5%" },
    daily: { value: 256, change: "+8.2%" },
    avgTime: { value: "2.4m", change: "-15.3%" },
    accuracy: { value: "96%", change: "+2.1%" }
  };

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8 min-h-[400px] items-center justify-center">
        <div className="glass rounded-xl p-10 text-center col-span-full">
          <Loader2 className="h-10 w-10 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading analytics data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 mb-8">
      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div 
          className={`glass rounded-lg p-5 cursor-pointer transition-all ${activeMetric === 'total' ? 'ring-2 ring-primary/50' : 'hover:bg-muted/30'}`}
          onClick={() => setActiveMetric('total')}
        >
          <div className="flex justify-between items-start mb-2">
            <div className="text-sm text-muted-foreground">Total Videos</div>
            <div className={`text-xs px-2 py-1 rounded-full ${metricStats.total.change.startsWith('+') ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
              {metricStats.total.change}
            </div>
          </div>
          <div className="text-3xl font-bold">{metricStats.total.value.toLocaleString()}</div>
        </div>
        
        <div 
          className={`glass rounded-lg p-5 cursor-pointer transition-all ${activeMetric === 'daily' ? 'ring-2 ring-primary/50' : 'hover:bg-muted/30'}`}
          onClick={() => setActiveMetric('daily')}
        >
          <div className="flex justify-between items-start mb-2">
            <div className="text-sm text-muted-foreground">Daily Average</div>
            <div className={`text-xs px-2 py-1 rounded-full ${metricStats.daily.change.startsWith('+') ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
              {metricStats.daily.change}
            </div>
          </div>
          <div className="text-3xl font-bold">{metricStats.daily.value}</div>
        </div>
        
        <div 
          className={`glass rounded-lg p-5 cursor-pointer transition-all ${activeMetric === 'avgTime' ? 'ring-2 ring-primary/50' : 'hover:bg-muted/30'}`}
          onClick={() => setActiveMetric('avgTime')}
        >
          <div className="flex justify-between items-start mb-2">
            <div className="text-sm text-muted-foreground">Avg. Processing</div>
            <div className={`text-xs px-2 py-1 rounded-full ${metricStats.avgTime.change.startsWith('-') ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
              {metricStats.avgTime.change}
            </div>
          </div>
          <div className="text-3xl font-bold">{metricStats.avgTime.value}</div>
        </div>
        
        <div 
          className={`glass rounded-lg p-5 cursor-pointer transition-all ${activeMetric === 'accuracy' ? 'ring-2 ring-primary/50' : 'hover:bg-muted/30'}`}
          onClick={() => setActiveMetric('accuracy')}
        >
          <div className="flex justify-between items-start mb-2">
            <div className="text-sm text-muted-foreground">Accuracy</div>
            <div className={`text-xs px-2 py-1 rounded-full ${metricStats.accuracy.change.startsWith('+') ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
              {metricStats.accuracy.change}
            </div>
          </div>
          <div className="text-3xl font-bold">{metricStats.accuracy.value}</div>
        </div>
      </div>
      
      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
          <div className="flex justify-center mt-2 text-sm text-muted-foreground">
            <div className="flex items-center mr-4">
              <div className="w-3 h-3 rounded-full bg-primary mr-2"></div>
              <span>45% Authentic</span>
            </div>
            <div className="flex items-center mr-4">
              <div className="w-3 h-3 rounded-full mr-2" style={{backgroundColor: 'hsl(350, 100%, 60%)'}}></div>
              <span>35% Deepfake</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full mr-2" style={{backgroundColor: 'hsl(45, 100%, 50%)'}}></div>
              <span>20% Suspicious</span>
            </div>
          </div>
        </div>
        
        {/* System Performance */}
        <div className="glass rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">System Performance</h2>
          <div className="h-64">
            <canvas ref={performanceChartRef}></canvas>
          </div>
        </div>
        
        {/* Accuracy Trends */}
        <div className="glass rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Accuracy Trends</h2>
          <div className="h-64">
            <canvas ref={trendsChartRef}></canvas>
          </div>
        </div>
        
        {/* Regional Insights */}
        <div className="glass rounded-xl p-6 col-span-full">
          <h2 className="text-xl font-semibold mb-4">Regional Insights</h2>
          <div className="h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-xl font-bold mb-1">North America</div>
                  <div className="text-sm text-muted-foreground">8,642 uploads</div>
                  <div className="text-xs text-green-500">38% of total</div>
                </div>
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-xl font-bold mb-1">Europe</div>
                  <div className="text-sm text-muted-foreground">6,125 uploads</div>
                  <div className="text-xs text-green-500">27% of total</div>
                </div>
                <div className="text-center p-3 bg-muted/20 rounded-lg">
                  <div className="text-xl font-bold mb-1">Asia Pacific</div>
                  <div className="text-sm text-muted-foreground">5,270 uploads</div>
                  <div className="text-xs text-green-500">24% of total</div>
                </div>
              </div>
              
              <div className="flex items-center justify-center">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  width="80" 
                  height="80" 
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
                <p className="text-sm text-muted-foreground ml-4">
                  Interactive world map visualization is being developed to show real-time usage patterns and hotspots for deepfake detection activities.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
