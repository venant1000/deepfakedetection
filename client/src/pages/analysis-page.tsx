import { useParams, useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import AnalysisSummary from "@/components/analysis/analysis-summary";
import VideoAnalysis from "@/components/analysis/video-analysis";
import DeepfakeDetails from "@/components/analysis/deepfake-details";

export default function AnalysisPage() {
  const { id } = useParams();
  const [, navigate] = useLocation();

  // For now, we're using a mock analysis (in a real app, we'd fetch this data)
  const analysis = {
    id,
    fileName: "interview_ceo.mp4",
    duration: "3:42",
    date: "June 15, 2023",
    isDeepfake: true,
    confidence: 94,
    issues: [
      {
        type: "error",
        text: "Inconsistent lip synchronization throughout the video (1:24-2:15)"
      },
      {
        type: "error",
        text: "Unnatural eye blinking patterns (too infrequent)"
      },
      {
        type: "warning",
        text: "Lighting inconsistencies on facial features"
      },
      {
        type: "warning",
        text: "Audio modulation artifacts detected at 1:45-2:02"
      }
    ],
    findings: [
      {
        title: "Lip Synchronization Issues",
        icon: "comment-alt",
        severity: "High",
        timespan: "1:24-2:15",
        description: "The subject's lip movements do not properly align with the spoken audio, particularly during complex phonetic sequences. This is a common indicator of audio replacement or visual manipulation."
      },
      {
        title: "Unnatural Eye Blinking",
        icon: "eye",
        severity: "High",
        timespan: "throughout video",
        description: "The average human blinks 15-20 times per minute, but the subject in this video blinks only 4 times per minute. This unnaturally low frequency is typical of earlier generation deepfakes."
      },
      {
        title: "Lighting Inconsistencies",
        icon: "lightbulb",
        severity: "Medium",
        timespan: "2:25-2:40",
        description: "Shadow directions on the face don't match the ambient lighting in the room. Particularly noticeable at 2:25 when the subject turns slightly but facial highlights remain static."
      },
      {
        title: "Audio Artifacts",
        icon: "volume-up",
        severity: "Medium",
        timespan: "1:45-2:02",
        description: "Spectral analysis reveals unnatural audio patterns consistent with voice synthesis. Specific frequency bands show machine learning artifacts typical of voice cloning technology."
      }
    ],
    timeline: [
      { position: 15, tooltip: "Eye blinking anomaly (0:33)", type: "normal" },
      { position: 28, tooltip: "Audio glitch (1:02)", type: "warning" },
      { position: 38, tooltip: "Major lip sync error (1:24)", type: "danger" },
      { position: 52, tooltip: "Facial texture inconsistency (1:56)", type: "danger" },
      { position: 65, tooltip: "Lighting anomaly (2:25)", type: "warning" },
      { position: 85, tooltip: "Unnatural head movement (3:10)", type: "normal" }
    ]
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Analysis Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <button 
                onClick={() => navigate("/dashboard")}
                className="text-muted-foreground hover:text-white transition-colors"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 19-7-7 7-7"/><path d="M19 12H5"/></svg>
              </button>
              <h1 className="text-2xl font-bold">Analysis Results</h1>
            </div>
            <p className="text-muted-foreground">{analysis.fileName} • {analysis.duration} minutes • Analyzed {analysis.date}</p>
          </div>
          
          <div className="flex items-center gap-3">
            <button className="py-2 px-4 rounded-lg glass-dark text-white hover:bg-muted transition-colors flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><path d="m16 6-4-4-4 4"/><path d="M12 2v13"/></svg>
              <span>Share</span>
            </button>
            <button className="py-2 px-4 rounded-lg bg-primary text-black font-medium hover:opacity-90 transition-all flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              <span>Download Report</span>
            </button>
          </div>
        </div>
        
        <AnalysisSummary analysis={analysis} />
        <VideoAnalysis analysis={analysis} />
        <DeepfakeDetails findings={analysis.findings} />
      </div>
    </div>
  );
}
