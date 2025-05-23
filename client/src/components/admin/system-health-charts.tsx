import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import { Progress } from "@/components/ui/progress";
import { Activity, Clock, Zap } from "lucide-react";

interface SystemHealthChartsProps {
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
  systemMetrics: {
    name: string;
    value: number;
    target: number;
    color: string;
  }[];
}

export default function SystemHealthCharts({ analyticsData, colors, systemMetrics }: SystemHealthChartsProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold mb-6">System Health Monitoring</h2>
      
      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <Card className="border-0 shadow-md">
          <CardHeader className="pb-2">
            <div className="flex justify-between items-start">
              <CardDescription>System Health</CardDescription>
              <span className="bg-primary/20 text-primary rounded-full p-1">
                <Activity className="h-4 w-4" />
              </span>
            </div>
            <CardTitle className="text-3xl font-bold">{analyticsData?.summary?.systemHealth || 0}%</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={analyticsData?.summary?.systemHealth || 0} className="h-2 mb-2" />
            <div className="text-xs text-muted-foreground">
              Overall system performance and stability
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-0 shadow-md">
          <CardHeader className="pb-2">
            <div className="flex justify-between items-start">
              <CardDescription>Analysis Speed</CardDescription>
              <span className="bg-secondary/20 text-secondary rounded-full p-1">
                <Clock className="h-4 w-4" />
              </span>
            </div>
            <CardTitle className="text-3xl font-bold">2.4s</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={80} className="h-2 mb-2" />
            <div className="text-xs text-muted-foreground">
              Average video frame analysis time
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-0 shadow-md">
          <CardHeader className="pb-2">
            <div className="flex justify-between items-start">
              <CardDescription>API Success Rate</CardDescription>
              <span className="bg-tertiary/20 text-tertiary rounded-full p-1">
                <Zap className="h-4 w-4" />
              </span>
            </div>
            <CardTitle className="text-3xl font-bold">99.7%</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={99.7} className="h-2 mb-2" />
            <div className="text-xs text-muted-foreground">
              Successful API requests last 24h
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-0 shadow-md">
          <CardHeader className="pb-2">
            <div className="flex justify-between items-start">
              <CardDescription>System Load</CardDescription>
              <span className="bg-quaternary/20 text-quaternary rounded-full p-1">
                <Activity className="h-4 w-4" />
              </span>
            </div>
            <CardTitle className="text-3xl font-bold">42%</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={42} className="h-2 mb-2" />
            <div className="text-xs text-muted-foreground">
              Current server resource utilization
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Resource Usage Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle>Resource Usage</CardTitle>
            <CardDescription>Real-time system resource monitoring</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-80 p-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={systemMetrics}
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 40, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#444" />
                  <XAxis type="number" domain={[0, 100]} tick={{ fill: '#888' }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#888' }} width={100} />
                  <Tooltip 
                    contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                    labelStyle={{ color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Bar dataKey="value" name="Current Usage (%)" radius={[0, 4, 4, 0]}>
                    {systemMetrics.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                  <ReferenceLine x={60} stroke={colors.warning} strokeDasharray="3 3" label={{ value: 'Warning', position: 'insideBottomRight', fill: colors.warning, fontSize: 12 }} />
                  <ReferenceLine x={80} stroke={colors.error} strokeDasharray="3 3" label={{ value: 'Critical', position: 'insideBottomRight', fill: colors.error, fontSize: 12 }} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle>Performance Over Time</CardTitle>
            <CardDescription>System response times (last 24 hours)</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-80 p-6">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={[
                    { time: '00:00', api: 120, db: 80, analysis: 320 },
                    { time: '03:00', api: 145, db: 92, analysis: 384 },
                    { time: '06:00', api: 100, db: 78, analysis: 298 },
                    { time: '09:00', api: 190, db: 140, analysis: 490 },
                    { time: '12:00', api: 220, db: 160, analysis: 580 },
                    { time: '15:00', api: 170, db: 120, analysis: 390 },
                    { time: '18:00', api: 150, db: 90, analysis: 340 },
                    { time: '21:00', api: 135, db: 85, analysis: 310 }
                  ]}
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#444" />
                  <XAxis dataKey="time" tick={{ fill: '#888' }} />
                  <YAxis tick={{ fill: '#888' }} />
                  <Tooltip 
                    contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                    labelStyle={{ color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="api" name="API Response (ms)" stroke={colors.primary} strokeWidth={2} dot={{ fill: colors.primary, r: 4 }} />
                  <Line type="monotone" dataKey="db" name="Database Query (ms)" stroke={colors.secondary} strokeWidth={2} dot={{ fill: colors.secondary, r: 4 }} />
                  <Line type="monotone" dataKey="analysis" name="Analysis Time (ms)" stroke={colors.quaternary} strokeWidth={2} dot={{ fill: colors.quaternary, r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}