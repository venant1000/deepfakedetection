interface AnalysisSummaryProps {
  analysis: {
    isDeepfake: boolean;
    confidence: number;
    issues: {
      type: string;
      text: string;
    }[];
    findings?: {
      title: string;
      icon: string;
      severity: string;
      timespan: string;
      description: string;
    }[];
  };
}

export default function AnalysisSummary({ analysis }: AnalysisSummaryProps) {

  return (
    <div className="glass rounded-xl p-6 mb-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <h2 className="text-xl font-semibold mb-4">Analysis Summary</h2>
          
          {/* Status indicator - different for deepfake vs authentic */}
          <div className={`p-4 rounded-lg ${analysis.isDeepfake ? 'bg-[#ff3366]/10 border border-[#ff3366]/20' : 'bg-primary/10 border border-primary/20'} mb-6`}>
            <div className="flex items-start gap-3">
              <div className={`${analysis.isDeepfake ? 'text-[#ff3366]' : 'text-primary'} mt-1`}>
                {analysis.isDeepfake ? (
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                )}
              </div>
              <div>
                <h3 className={`font-semibold ${analysis.isDeepfake ? 'text-[#ff3366]' : 'text-primary'}`}>
                  {analysis.isDeepfake 
                    ? `Deepfake Detected (${Math.round(analysis.confidence * 100)}% Confidence)` 
                    : `Content Appears Authentic (${Math.round(analysis.confidence * 100)}% Confidence)`}
                </h3>
                <p className="text-muted-foreground mt-1">
                  {analysis.isDeepfake 
                    ? "Our AI has identified significant manipulations in this video that indicate it is likely a deepfake." 
                    : "Our AI analysis indicates this content appears to be authentic with no significant signs of manipulation."}
                </p>
              </div>
            </div>
          </div>
          
          {/* Issues section */}
          {analysis.issues && analysis.issues.length > 0 && (
            <div className="mb-6">
              <h3 className="font-medium mb-3">{analysis.isDeepfake ? "Key Issues Detected:" : "Analysis Notes:"}</h3>
              <ul className="space-y-2 text-muted-foreground">
                {analysis.issues.map((issue, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <svg 
                      xmlns="http://www.w3.org/2000/svg" 
                      width="12" 
                      height="12" 
                      viewBox="0 0 24 24" 
                      fill="currentColor" 
                      className={`mt-1.5 ${
                        issue.type === 'error' 
                          ? 'text-[#ff3366]' 
                          : issue.type === 'warning' 
                          ? 'text-[#ffbb00]' 
                          : 'text-primary'
                      }`}
                    >
                      <circle cx="12" cy="12" r="12" />
                    </svg>
                    <span>{issue.text}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          <div>
            <h3 className="font-medium mb-3">Analysis Methodology:</h3>
            <p className="text-muted-foreground mb-2">
              This video was analyzed using Google's Gemini AI with DeepGuard's proprietary algorithms focusing on:
            </p>
            <ul className="space-y-1 text-muted-foreground">
              {[
                "Facial integrity and movement consistency",
                "Audio-visual synchronization",
                "Digital artifact detection",
                "Background consistency analysis"
              ].map((item, index) => (
                <li key={index} className="flex items-start gap-2">
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    width="12" 
                    height="12" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="3" 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    className="text-primary mt-1.5"
                  >
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        
        <div className="glass-dark rounded-xl p-6 flex flex-col">
          <h3 className="text-lg font-semibold mb-6 text-center">Confidence Score</h3>
          
          <div className="relative mx-auto h-44 w-44 mb-6">
            <svg className="w-full h-full" viewBox="0 0 100 100">
              <circle className="text-muted" strokeWidth="10" stroke="currentColor" fill="transparent" r="40" cx="50" cy="50"/>
              <circle 
                className={`${analysis.isDeepfake ? 'text-[#ff3366]' : 'text-primary'} progress-ring`} 
                strokeWidth="10" 
                stroke="currentColor" 
                fill="transparent" 
                r="40" 
                cx="50" 
                cy="50" 
                strokeDasharray="251.2" 
                strokeDashoffset={(100 - (analysis.confidence * 100)) / 100 * 251.2}
              />
            </svg>
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center">
              <div className="text-4xl font-bold">{Math.round(analysis.confidence * 100)}%</div>
              <div className="text-sm text-muted-foreground">{analysis.isDeepfake ? "Deepfake" : "Authentic"}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
