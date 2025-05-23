import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
  Cell
} from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import { Badge } from "@/components/ui/badge";

interface ModelPerformanceChartsProps {
  analyticsData: any;
  colors: {
    primary: string;
    secondary: string;
    tertiary: string;
    quaternary: string;
    warning: string;
    neutral: string;
    success: string;
    error: string;
    dark: string;
    light: string;
  };
  modelPerformanceData: {
    name: string;
    current: number;
    previous: number;
  }[];
}

export default function ModelPerformanceCharts({ analyticsData, colors, modelPerformanceData }: ModelPerformanceChartsProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold mb-6">ML Model Performance</h2>
      
      {/* Model Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {modelPerformanceData.map((metric, index) => (
          <Card key={index} className="border-0 shadow-md">
            <CardHeader className="pb-2">
              <div className="flex justify-between items-start">
                <CardDescription>{metric.name}</CardDescription>
                <span className={`bg-primary/20 text-primary rounded-full p-1 ${
                  metric.current > metric.previous ? "text-green-500" : "text-red-500"
                }`}>
                  {metric.current > metric.previous ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m18 15-6-6-6 6"/></svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6"/></svg>
                  )}
                </span>
              </div>
              <CardTitle className="text-3xl font-bold">{metric.current}%</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center">
                <div className="text-xs text-muted-foreground flex items-center">
                  {metric.current > metric.previous ? (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-500 mr-1"><path d="m18 15-6-6-6 6"/></svg>
                      <span className="text-green-500">+{(metric.current - metric.previous).toFixed(1)}%</span>
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-red-500 mr-1"><path d="m6 9 6 6 6-6"/></svg>
                      <span className="text-red-500">-{(metric.previous - metric.current).toFixed(1)}%</span>
                    </>
                  )}
                  <span className="ml-1">from previous version</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      
      {/* Confidence Score vs. Accuracy Chart */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle>Model Confidence vs. Accuracy</CardTitle>
            <CardDescription>Correlation between confidence scores and actual accuracy</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-80 p-6">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis type="number" dataKey="confidence" name="Confidence Score" domain={[0, 100]} unit="%" tick={{ fill: '#888' }} />
                  <YAxis type="number" dataKey="accuracy" name="Actual Accuracy" domain={[0, 100]} unit="%" tick={{ fill: '#888' }} />
                  <Tooltip 
                    contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                    labelStyle={{ color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Scatter 
                    name="Deepfake Videos" 
                    data={[
                      { confidence: 92, accuracy: 95, count: 15 },
                      { confidence: 86, accuracy: 89, count: 23 },
                      { confidence: 78, accuracy: 82, count: 18 },
                      { confidence: 65, accuracy: 72, count: 12 },
                      { confidence: 58, accuracy: 62, count: 9 },
                      { confidence: 45, accuracy: 48, count: 5 },
                    ]} 
                    fill={colors.quaternary}
                  />
                  <Scatter 
                    name="Authentic Videos" 
                    data={[
                      { confidence: 95, accuracy: 97, count: 20 },
                      { confidence: 88, accuracy: 92, count: 28 },
                      { confidence: 81, accuracy: 85, count: 22 },
                      { confidence: 73, accuracy: 78, count: 17 },
                      { confidence: 62, accuracy: 68, count: 10 },
                      { confidence: 52, accuracy: 55, count: 7 },
                    ]} 
                    fill={colors.primary}
                  />
                  <ReferenceLine y={75} stroke={colors.warning} strokeDasharray="3 3" label={{ value: 'Minimum Target', position: 'insideBottomRight', fill: colors.warning, fontSize: 12 }} />
                  <ReferenceLine x={75} stroke={colors.warning} strokeDasharray="3 3" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        {/* Model Performance Trend */}
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle>Performance Trend</CardTitle>
            <CardDescription>Model accuracy over version iterations</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-80 p-6">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={[
                    { version: "v1.0", accuracy: 82.5, precision: 80.3, recall: 78.9, f1: 79.6 },
                    { version: "v1.2", accuracy: 85.7, precision: 83.4, recall: 82.2, f1: 82.8 },
                    { version: "v1.5", accuracy: 88.9, precision: 87.1, recall: 85.6, f1: 86.3 },
                    { version: "v1.8", accuracy: 91.2, precision: 90.5, recall: 88.7, f1: 89.6 },
                    { version: "v2.0", accuracy: 94.5, precision: 92.8, recall: 91.4, f1: 92.1 },
                    { version: "v2.2", accuracy: 96.3, precision: 94.7, recall: 93.8, f1: 94.2 }
                  ]}
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#444" />
                  <XAxis dataKey="version" tick={{ fill: '#888' }} />
                  <YAxis domain={[75, 100]} tick={{ fill: '#888' }} />
                  <Tooltip 
                    contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                    labelStyle={{ color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke={colors.primary} strokeWidth={3} dot={{ fill: colors.primary, r: 4 }} activeDot={{ r: 6 }} />
                  <Line type="monotone" dataKey="precision" name="Precision" stroke={colors.secondary} strokeWidth={2} dot={{ fill: colors.secondary, r: 4 }} />
                  <Line type="monotone" dataKey="recall" name="Recall" stroke={colors.tertiary} strokeWidth={2} dot={{ fill: colors.tertiary, r: 4 }} />
                  <Line type="monotone" dataKey="f1" name="F1 Score" stroke={colors.quaternary} strokeWidth={2} dot={{ fill: colors.quaternary, r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Confusion Matrix and Error Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Confusion Matrix Card */}
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle>Confusion Matrix</CardTitle>
            <CardDescription>Model prediction evaluation</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="glass p-4 rounded-md text-center">
                <div className="text-xs mb-2 text-muted-foreground">True Positive</div>
                <div className="text-2xl font-bold text-green-500">382</div>
                <div className="text-xs mt-2 text-muted-foreground">Correctly identified deepfakes</div>
              </div>
              <div className="glass p-4 rounded-md text-center">
                <div className="text-xs mb-2 text-muted-foreground">False Positive</div>
                <div className="text-2xl font-bold text-yellow-500">21</div>
                <div className="text-xs mt-2 text-muted-foreground">Authentic incorrectly flagged</div>
              </div>
              <div className="glass p-4 rounded-md text-center">
                <div className="text-xs mb-2 text-muted-foreground">False Negative</div>
                <div className="text-2xl font-bold text-red-500">18</div>
                <div className="text-xs mt-2 text-muted-foreground">Missed deepfakes</div>
              </div>
              <div className="glass p-4 rounded-md text-center">
                <div className="text-xs mb-2 text-muted-foreground">True Negative</div>
                <div className="text-2xl font-bold text-blue-500">579</div>
                <div className="text-xs mt-2 text-muted-foreground">Correctly identified authentic</div>
              </div>
            </div>
            <div className="text-sm text-center text-muted-foreground">
              Model evaluation based on the latest 1,000 processed videos
            </div>
          </CardContent>
        </Card>
        
        {/* Error Cases Analysis */}
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle>Error Case Analysis</CardTitle>
            <CardDescription>Top reasons for incorrect classifications</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-80 p-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={[
                    { reason: "Poor video quality", count: 14, type: "negative" },
                    { reason: "Advanced deepfake technique", count: 12, type: "negative" },
                    { reason: "Lighting inconsistencies", count: 8, type: "positive" },
                    { reason: "Unusual camera angles", count: 6, type: "positive" },
                    { reason: "Complex backgrounds", count: 5, type: "both" },
                    { reason: "Face partially obscured", count: 4, type: "both" }
                  ]}
                  margin={{ top: 20, right: 30, left: 140, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#444" />
                  <XAxis type="number" tick={{ fill: '#888' }} />
                  <YAxis dataKey="reason" type="category" tick={{ fill: '#888' }} width={140} />
                  <Tooltip 
                    contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                    labelStyle={{ color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Bar 
                    dataKey="count" 
                    name="Error Cases" 
                    radius={[0, 4, 4, 0]}
                    fill={colors.quaternary}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}