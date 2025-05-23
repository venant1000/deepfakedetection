import React from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area
} from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface ChartProps {
  analyticsData?: any;
}

// Chart colors
const colors = {
  primary: "#00ff88",
  secondary: "#7000ff",
  tertiary: "#00a3ff",
  quaternary: "#ff3e66",
  warning: "#ff9500"
};

const CHART_COLORS = [colors.primary, colors.secondary, colors.tertiary, colors.quaternary];

export const ConfidenceScoreChart: React.FC<ChartProps> = ({ analyticsData }) => {
  // Generate sample data if no analytics data is provided
  const confidenceData = [
    { range: "0-20%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.15) : 3 },
    { range: "20-40%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.25) : 5 },
    { range: "40-60%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.3) : 6 },
    { range: "60-80%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.2) : 4 },
    { range: "80-100%", count: analyticsData?.summary?.videoCount ? Math.floor(analyticsData.summary.videoCount * 0.1) : 2 }
  ];

  return (
    <Card className="border-0 shadow-md">
      <CardHeader>
        <CardTitle>Confidence Score Distribution</CardTitle>
        <CardDescription>Distribution of detection confidence levels</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-80 p-6">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={confidenceData}
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#444" />
              <XAxis dataKey="range" tick={{ fill: '#888' }} />
              <YAxis tick={{ fill: '#888' }} />
              <Tooltip 
                contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                labelStyle={{ color: '#fff' }}
                itemStyle={{ color: '#fff' }}
              />
              <Bar 
                dataKey="count" 
                name="Video Count" 
                fill={colors.primary}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export const DeepfakeCategoriesChart: React.FC<ChartProps> = ({ analyticsData }) => {
  const categoriesData = analyticsData?.detectionTypes || [
    { name: "Facial Manipulation", value: 45 },
    { name: "Voice Synthesis", value: 20 },
    { name: "Body Movements", value: 15 },
    { name: "Background Alterations", value: 10 }
  ];

  return (
    <Card className="border-0 shadow-md">
      <CardHeader>
        <CardTitle>Deepfake Categories</CardTitle>
        <CardDescription>Distribution by manipulation type</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-80 p-6">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={categoriesData}
                nameKey="name"
                dataKey="value"
                cx="50%"
                cy="50%"
                outerRadius="70%"
                innerRadius="40%"
                paddingAngle={2}
                label={(entry) => entry.name}
                labelLine={{ stroke: "#555" }}
              >
                {categoriesData.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                labelStyle={{ color: '#fff' }}
                itemStyle={{ color: '#fff' }}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export const DetectionRatesChart: React.FC<ChartProps> = ({ analyticsData }) => {
  const ratesData = analyticsData?.detectionRates || [
    { date: "May 17", rate: 23.5 },
    { date: "May 18", rate: 24.8 },
    { date: "May 19", rate: 28.4 },
    { date: "May 20", rate: 26.2 },
    { date: "May 21", rate: 29.5 },
    { date: "May 22", rate: 33.1 },
    { date: "May 23", rate: 30.8 }
  ];

  return (
    <Card className="border-0 shadow-md">
      <CardHeader>
        <CardTitle>Detection Rate Trends</CardTitle>
        <CardDescription>Deepfake detection rates over time</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-80 p-6">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={ratesData}
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#444" />
              <XAxis dataKey="date" tick={{ fill: '#888' }} />
              <YAxis tick={{ fill: '#888' }} />
              <Tooltip 
                contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                labelStyle={{ color: '#fff' }}
                itemStyle={{ color: '#fff' }}
              />
              <Line 
                type="monotone" 
                dataKey="rate" 
                name="Detection Rate (%)" 
                stroke={colors.quaternary} 
                strokeWidth={3}
                dot={{ fill: colors.quaternary, r: 4 }}
                activeDot={{ fill: colors.quaternary, r: 6, stroke: 'white', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export const ProcessingTimeChart: React.FC<ChartProps> = ({ analyticsData }) => {
  const processingData = analyticsData?.processingTimes || [
    { timeRange: "<30s", count: 45 },
    { timeRange: "30s-1m", count: 32 },
    { timeRange: "1m-2m", count: 18 },
    { timeRange: "2m-5m", count: 5 },
    { timeRange: ">5m", count: 2 }
  ];

  return (
    <Card className="border-0 shadow-md">
      <CardHeader>
        <CardTitle>Processing Time Distribution</CardTitle>
        <CardDescription>Analysis time for video processing</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-80 p-6">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={processingData}
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#444" />
              <XAxis dataKey="timeRange" tick={{ fill: '#888' }} />
              <YAxis tick={{ fill: '#888' }} />
              <Tooltip 
                contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                labelStyle={{ color: '#fff' }}
                itemStyle={{ color: '#fff' }}
              />
              <Bar 
                dataKey="count" 
                name="Video Count" 
                fill={colors.tertiary}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export const ClassificationBreakdownChart: React.FC<ChartProps> = ({ analyticsData }) => {
  const breakdownData = analyticsData?.classificationBreakdown || [
    { name: "Authentic", value: 65, color: "#00ff88" },
    { name: "Deepfake", value: 25, color: "#ff3366" },
    { name: "Moderate/Suspicious", value: 10, color: "#ffaa00" }
  ];

  return (
    <Card className="border-0 shadow-md">
      <CardHeader>
        <CardTitle>Classification Breakdown</CardTitle>
        <CardDescription>Video classification results</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-80 p-6">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={breakdownData}
                nameKey="name"
                dataKey="value"
                cx="50%"
                cy="50%"
                outerRadius="70%"
                innerRadius={0}
                paddingAngle={0}
                label={(entry) => `${entry.name}: ${entry.value}%`}
              >
                {breakdownData.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: '6px' }}
                labelStyle={{ color: '#fff' }}
                itemStyle={{ color: '#fff' }}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};