interface Finding {
  title: string;
  icon: string;
  severity: string;
  timespan: string;
  description: string;
}

interface DeepfakeDetailsProps {
  findings: Finding[];
}

export default function DeepfakeDetails({ findings }: DeepfakeDetailsProps) {
  // Map icons to their SVG representation
  const getIcon = (iconName: string) => {
    switch (iconName) {
      case 'comment-alt':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        );
      case 'eye':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        );
      case 'lightbulb':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 18h6" />
            <path d="M10 22h4" />
            <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14" />
          </svg>
        );
      case 'volume-up':
        return (
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
          </svg>
        );
      default:
        return (
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 16v-4" />
            <path d="M12 8h.01" />
          </svg>
        );
    }
  };

  // Map severity to color
  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high':
        return 'text-[#ff3366]';
      case 'medium':
        return 'text-[#ffbb00]';
      case 'low':
        return 'text-primary';
      default:
        return 'text-muted-foreground';
    }
  };

  return (
    <div className="glass rounded-xl p-6 mb-8">
      <h2 className="text-xl font-semibold mb-6">Detailed Findings</h2>
      
      <div className="space-y-6">
        {findings && findings.length > 0 ? (
          findings.map((finding, index) => (
            <div key={index} className="glass-dark rounded-lg p-4">
              <h3 className="font-medium text-lg mb-2 flex items-center gap-2">
                <span className={getSeverityColor(finding.severity)}>
                  {getIcon(finding.icon)}
                </span>
                <span>{finding.title}</span>
              </h3>
              <p className="text-muted-foreground mb-3">{finding.description}</p>
              <div className="text-sm text-muted-foreground">
                <span className={`${getSeverityColor(finding.severity)} font-medium`}>
                  Severity: {finding.severity}
                </span> • Detected {finding.timespan && finding.timespan.includes('throughout') ? 'throughout video' : finding.timespan ? `between ${finding.timespan}` : 'in the video'}
              </div>
            </div>
          ))
        ) : (
          <div className="glass-dark rounded-lg p-6 text-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-4 text-muted-foreground">
              <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"/>
              <path d="M12 16v-4"/>
              <path d="M12 8h.01"/>
            </svg>
            <h3 className="text-lg font-medium mb-2">No Detailed Findings Available</h3>
            <p className="text-muted-foreground">
              The analysis didn't provide specific detailed findings for this video. 
              This might happen with videos that have subtle manipulations or if the system 
              determined a confidence score based on overall patterns rather than specific issues.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
