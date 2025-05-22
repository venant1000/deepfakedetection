import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { Loader2 } from "lucide-react";

export default function AdminStats() {
  const { user } = useAuth();
  const [isLoading, setIsLoading] = useState(true);
  const [statsData, setStatsData] = useState({
    totalUsers: 0,
    videoCount: 0,
    deepfakesDetected: 0,
    systemHealth: 100
  });

  // Fetch real data from the database
  useEffect(() => {
    const fetchStats = async () => {
      try {
        setIsLoading(true);
        const response = await fetch("/api/admin/stats", {
          credentials: "include",
        });

        if (response.ok) {
          const data = await response.json();
          setStatsData({
            totalUsers: data.summary.totalUsers,
            videoCount: data.summary.videoCount,
            deepfakesDetected: data.summary.deepfakesDetected,
            systemHealth: data.summary.systemHealth
          });
        }
      } catch (error) {
        console.error("Failed to fetch admin stats:", error);
      } finally {
        setIsLoading(false);
      }
    };

    if (user?.username?.includes("admin")) {
      fetchStats();
    }
  }, [user]);

  const stats = [
    {
      title: "Total Users",
      value: isLoading ? "..." : statsData.totalUsers.toLocaleString(),
      change: "+8.2%",
      isPositive: true,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
          <circle cx="9" cy="7" r="4" />
          <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
          <path d="M16 3.13a4 4 0 0 1 0 7.75" />
        </svg>
      ),
      bgColor: "bg-blue-500/20",
      textColor: "text-blue-400"
    },
    {
      title: "Videos Analyzed",
      value: isLoading ? "..." : statsData.videoCount.toLocaleString(),
      change: "+12.4%",
      isPositive: true,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="m22 8-6-6H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <path d="M14 2v6h6" />
          <path d="M10 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z" />
          <path d="m22 16-5.23-5.23a1 1 0 0 0-1.41 0L12 14.12l-1.36-1.36a1 1 0 0 0-1.41 0L2 20" />
        </svg>
      ),
      bgColor: "bg-purple-500/20",
      textColor: "text-purple-400"
    },
    {
      title: "Deepfakes Detected",
      value: isLoading ? "..." : statsData.deepfakesDetected.toLocaleString(),
      change: "+17.9%",
      isPositive: false,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
          <line x1="12" x2="12" y1="9" y2="13" />
          <line x1="12" x2="12.01" y1="17" y2="17" />
        </svg>
      ),
      bgColor: "bg-red-500/20",
      textColor: "text-red-400"
    },
    {
      title: "System Health",
      value: isLoading ? "..." : `${statsData.systemHealth}%`,
      change: "+0.5%",
      isPositive: true,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
          <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
          <line x1="6" y1="6" x2="6.01" y2="6" />
          <line x1="6" y1="18" x2="6.01" y2="18" />
        </svg>
      ),
      bgColor: "bg-green-500/20",
      textColor: "text-green-400"
    }
  ];

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
            <span className={stat.isPositive ? "text-primary mr-1" : "text-[#ff3366] mr-1"}>
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                className="inline"
              >
                {stat.isPositive ? (
                  <path d="m18 15-6-6-6 6" />
                ) : (
                  <path d="m6 9 6 6 6-6" />
                )}
              </svg> {stat.change}
            </span>
            <span className="text-muted-foreground">
              {index === 3 ? "from last week" : "from last month"}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
