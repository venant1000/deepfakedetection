import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";

// In a real app, we would fetch this data from the API
const recentAnalyses = [
  {
    id: '1',
    fileName: 'interview_ceo.mp4',
    date: 'Today, 10:34 AM',
    duration: '3:42',
    result: { status: 'deepfake', confidence: 94 }
  },
  {
    id: '2',
    fileName: 'product_demo.mp4',
    date: 'Yesterday, 2:15 PM',
    duration: '2:18',
    result: { status: 'authentic', confidence: 99 }
  },
  {
    id: '3',
    fileName: 'social_ad.mp4',
    date: 'Jun 12, 8:22 AM',
    duration: '0:45',
    result: { status: 'suspicious', confidence: 67 }
  }
];

export default function RecentAnalyses() {
  const [, navigate] = useLocation();
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deepfake':
        return 'bg-[#ff3366]/20 text-[#ff3366]';
      case 'authentic':
        return 'bg-primary/20 text-primary';
      case 'suspicious':
        return 'bg-[#ffbb00]/20 text-[#ffbb00]';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };
  
  return (
    <div className="glass rounded-xl p-6 mb-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold">Recent Analyses</h2>
        <a href="#view-all" className="text-primary text-sm hover:underline">View All</a>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-muted">
              <th className="pb-3 text-left text-muted-foreground font-medium">File Name</th>
              <th className="pb-3 text-left text-muted-foreground font-medium">Date</th>
              <th className="pb-3 text-left text-muted-foreground font-medium">Duration</th>
              <th className="pb-3 text-left text-muted-foreground font-medium">Result</th>
              <th className="pb-3 text-left text-muted-foreground font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {recentAnalyses.map((analysis) => (
              <tr key={analysis.id} className="border-b border-muted">
                <td className="py-4">
                  <div className="flex items-center gap-3">
                    <div className="h-9 w-9 rounded bg-muted flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground"><path d="m22 8-6-6H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M10 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/><path d="m22 16-5.23-5.23a1 1 0 0 0-1.41 0L12 14.12l-1.36-1.36a1 1 0 0 0-1.41 0L2 20"/></svg>
                    </div>
                    <span>{analysis.fileName}</span>
                  </div>
                </td>
                <td className="py-4 text-muted-foreground">{analysis.date}</td>
                <td className="py-4 text-muted-foreground">{analysis.duration}</td>
                <td className="py-4">
                  <span className={`py-1 px-3 rounded-full text-sm ${getStatusColor(analysis.result.status)}`}>
                    {analysis.result.status.charAt(0).toUpperCase() + analysis.result.status.slice(1)} ({analysis.result.confidence}%)
                  </span>
                </td>
                <td className="py-4">
                  <div className="flex items-center gap-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-muted-foreground hover:text-foreground transition-colors"
                      onClick={() => navigate(`/analysis/${analysis.id}`)}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
