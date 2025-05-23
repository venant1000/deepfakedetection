import { useEffect, useRef, useState } from "react";
import Chart from "chart.js/auto";

interface AnalyticsDashboardProps {
  analyticsData?: any;
}

export default function AnalyticsDashboard({ analyticsData }: AnalyticsDashboardProps) {
  const usageChartRef = useRef<HTMLCanvasElement | null>(null);
  const detectionChartRef = useRef<HTMLCanvasElement | null>(null);
  const performanceChartRef = useRef<HTMLCanvasElement | null>(null);
  const trendsChartRef = useRef<HTMLCanvasElement | null>(null);
  
  const usageChartInstance = useRef<Chart | null>(null);
  const detectionChartInstance = useRef<Chart | null>(null);
  const performanceChartInstance = useRef<Chart | null>(null);
  const trendsChartInstance = useRef<Chart | null>(null);

  const [activeMetric, setActiveMetric] = useState("total");
  
  useEffect(() => {
    // Only initialize charts if we have data
    if (analyticsData) {
      // Cleanup previous chart instances to prevent duplicates
      destroyCharts();
      // Initialize charts immediately
      initializeCharts();
    }
    
    // Cleanup when unmounting
    return () => {
      destroyCharts();
    };
  }, [analyticsData]);  // Re-initialize when data changes

  const destroyCharts = () => {
    if (usageChartInstance.current) {
      usageChartInstance.current.destroy();
      usageChartInstance.current = null;
    }
    if (detectionChartInstance.current) {
      detectionChartInstance.current.destroy();
      detectionChartInstance.current = null;
    }
    if (performanceChartInstance.current) {
      performanceChartInstance.current.destroy();
      performanceChartInstance.current = null;
    }
    if (trendsChartInstance.current) {
      trendsChartInstance.current.destroy();
      trendsChartInstance.current = null;
    }
  };

  const initializeCharts = () => {
    if (!analyticsData) return;

    // Prepare data for usage chart (daily uploads) - with fallback data
    if (usageChartRef.current) {
      const ctx = usageChartRef.current.getContext('2d');
      if (ctx) {
        // Check if we have actual data or if we need defaults
        const dailyUploads = analyticsData?.dailyUploads || [];
        // Create at least some data so charts aren't empty
        const labels = dailyUploads.length > 0 ? 
          dailyUploads.map((item: any) => item.date) : 
          ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
        const uploadCounts = dailyUploads.length > 0 ? 
          dailyUploads.map((item: any) => item.count) : 
          [analyticsData?.summary?.videoCount || 1, 0, 0, 0, 0, 0, 0];

        usageChartInstance.current = new Chart(ctx, {
          type: 'line',
          data: {
            labels,
            datasets: [{
              label: 'Videos Analyzed',
              data: uploadCounts,
              borderColor: 'hsl(210, 84%, 60%)',
              backgroundColor: 'hsla(210, 84%, 60%, 0.1)',
              tension: 0.3,
              fill: true
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
                backgroundColor: 'rgba(72, 85, 150, 0.9)',
                titleColor: 'white',
                bodyColor: 'white',
                borderColor: 'rgba(123, 97, 255, 0.3)',
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

    // Prepare detection results chart
    if (detectionChartRef.current) {
      const ctx = detectionChartRef.current.getContext('2d');
      if (ctx) {
        const totalVideos = analyticsData?.summary?.videoCount || 0;
        const totalDeepfakes = analyticsData?.summary?.deepfakeCount || 0;
        const authentic = totalVideos - totalDeepfakes;
        const deepfake = totalDeepfakes;
        
        detectionChartInstance.current = new Chart(ctx, {
          type: 'doughnut',
          data: {
            labels: ['Authentic', 'Deepfake'],
            datasets: [{
              data: [authentic, deepfake],
              backgroundColor: [
                'hsl(160, 84%, 39%)',
                'hsl(350, 84%, 60%)'
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
                borderWidth: 1
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

    // Prepare system performance chart (processing times)
    if (performanceChartRef.current) {
      const ctx = performanceChartRef.current.getContext('2d');
      if (ctx) {
        const processingTimes = analyticsData?.processingTimes || [];
        // Add default data if none exists
        const timeRanges = processingTimes.length > 0 ? 
          processingTimes.map((item: any) => item.timeRange) :
          ['<30s', '30s-1m', '1m-2m', '2m-5m', '>5m'];
        const counts = processingTimes.length > 0 ? 
          processingTimes.map((item: any) => item.count) :
          [analyticsData?.summary?.videoCount || 1, 0, 0, 0, 0];

        performanceChartInstance.current = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: timeRanges,
            datasets: [{
              label: 'Processing Time Distribution',
              data: counts,
              backgroundColor: 'hsla(250, 84%, 60%, 0.7)',
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
            },
            animation: {
              delay: (context) => context.dataIndex * 100
            }
          }
        });
      }
    }

    // Prepare detection rates trend chart
    if (trendsChartRef.current) {
      const ctx = trendsChartRef.current.getContext('2d');
      if (ctx) {
        const detectionRates = analyticsData?.detectionRates || [];
        
        // Add default data if none exists
        const dates = detectionRates.length > 0 ? 
          detectionRates.map((item: any) => item.date) :
          ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
          
        const rates = detectionRates.length > 0 ? 
          detectionRates.map((item: any) => item.rate) :
          [0, 0, 0, 0, 0, 0, analyticsData?.summary?.deepfakeCount && analyticsData?.summary?.videoCount ? 
            ((analyticsData.summary.deepfakeCount / analyticsData.summary.videoCount) * 100) : 0];

        trendsChartInstance.current = new Chart(ctx, {
          type: 'line',
          data: {
            labels: dates,
            datasets: [{
              label: 'Detection Rate',
              data: rates,
              borderColor: 'hsl(280, 84%, 60%)',
              backgroundColor: 'transparent',
              tension: 0.4,
              borderWidth: 3,
              pointBackgroundColor: 'hsl(280, 84%, 60%)',
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
                    return `Rate: ${context.raw}%`;
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

  // Calculate metrics from real data
  const videoCount = analyticsData?.summary?.videoCount || 0;
  const deepfakeCount = analyticsData?.summary?.deepfakeCount || 0;
  const systemHealth = analyticsData?.summary?.systemHealth || 0;
  
  // Get processing time average if available
  const processingTimes = analyticsData?.processingTimes || [];
  const hasProcessingData = processingTimes.length > 0 && processingTimes.some((pt: any) => pt.count > 0);

  const metricStats = {
    total: { 
      value: videoCount,
      change: "+100%" // This would be calculated based on previous period in a real app
    },
    deepfakes: { 
      value: deepfakeCount,

      change: deepfakeCount > 0 ? "+100%" : "0%" 
    },
    avgTime: { 
      value: hasProcessingData ? "~1m" : "N/A",
      change: "0%" 
    },
    accuracy: { 
      value: `${systemHealth}%`,
      change: "+0%" 
    }
  };

  // Show analytics immediately without loading state to prevent flickering

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
          className={`glass rounded-lg p-5 cursor-pointer transition-all ${activeMetric === 'deepfakes' ? 'ring-2 ring-primary/50' : 'hover:bg-muted/30'}`}
          onClick={() => setActiveMetric('deepfakes')}
        >
          <div className="flex justify-between items-start mb-2">
            <div className="text-sm text-muted-foreground">Deepfakes</div>
            <div className={`text-xs px-2 py-1 rounded-full ${metricStats.deepfakes.change.startsWith('+') ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
              {metricStats.deepfakes.change}
            </div>
          </div>
          <div className="text-3xl font-bold">{metricStats.deepfakes.value}</div>
        </div>
        
        <div 
          className={`glass rounded-lg p-5 cursor-pointer transition-all ${activeMetric === 'avgTime' ? 'ring-2 ring-primary/50' : 'hover:bg-muted/30'}`}
          onClick={() => setActiveMetric('avgTime')}
        >
          <div className="flex justify-between items-start mb-2">
            <div className="text-sm text-muted-foreground">Avg. Processing</div>
            <div className={`text-xs px-2 py-1 rounded-full bg-blue-500/10 text-blue-400`}>
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
            <div className="text-sm text-muted-foreground">System Health</div>
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
          <h2 className="text-xl font-semibold mb-4">Daily Upload Activity</h2>
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
              <span>Authentic: {videoCount - deepfakeCount}</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full mr-2" style={{backgroundColor: 'hsl(350, 100%, 60%)'}}></div>
              <span>Deepfake: {deepfakeCount}</span>
            </div>
          </div>
        </div>
        
        {/* Processing Time Distribution */}
        <div className="glass rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Processing Time Distribution</h2>
          <div className="h-64">
            <canvas ref={performanceChartRef}></canvas>
          </div>
        </div>
        
        {/* Detection Rate Trends */}
        <div className="glass rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Detection Rate Trends</h2>
          <div className="h-64">
            <canvas ref={trendsChartRef}></canvas>
          </div>
        </div>
      </div>
    </div>
  );
}