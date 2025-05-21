import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function ReportsPage() {
  const { user } = useAuth();
  const [searchTerm, setSearchTerm] = useState("");
  const [reportType, setReportType] = useState("all");
  
  // Mock data for demonstration
  const reports = [
    {
      id: "rep1",
      date: "2025-05-21",
      title: "Monthly Deepfake Statistics",
      type: "statistical",
      status: "completed"
    },
    {
      id: "rep2",
      date: "2025-05-18",
      title: "Detection Accuracy Report",
      type: "analytical",
      status: "completed"
    },
    {
      id: "rep3",
      date: "2025-05-15",
      title: "Common Manipulation Techniques",
      type: "educational",
      status: "completed"
    },
    {
      id: "rep4",
      date: "2025-05-12",
      title: "Deepfake Detection Technology Trends",
      type: "analytical",
      status: "completed"
    },
    {
      id: "rep5",
      date: "2025-05-10",
      title: "Weekly Analysis Summary",
      type: "statistical",
      status: "completed"
    },
    {
      id: "rep6",
      date: "2025-05-05",
      title: "User Activity Report",
      type: "statistical",
      status: "completed"
    }
  ];

  const savedReports = [
    {
      id: "saved1",
      date: "2025-04-30",
      title: "Media Forensics Techniques",
      type: "educational",
      saved: true
    },
    {
      id: "saved2",
      date: "2025-04-25",
      title: "AI Detection Algorithms Explained",
      type: "educational",
      saved: true
    },
    {
      id: "saved3",
      date: "2025-04-20",
      title: "Advanced Deepfake Identification",
      type: "educational",
      saved: true
    }
  ];

  // Get badge color based on report type
  const getTypeBadge = (type: string) => {
    switch (type) {
      case "statistical":
        return <Badge variant="secondary">{type}</Badge>;
      case "analytical":
        return <Badge variant="default" className="bg-blue-500 hover:bg-blue-600">{type}</Badge>;
      case "educational":
        return <Badge variant="default" className="bg-green-500 hover:bg-green-600">{type}</Badge>;
      default:
        return <Badge>{type}</Badge>;
    }
  };

  // Filter reports based on search term and type
  const filteredReports = reports.filter(report => {
    const matchesSearch = 
      report.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      report.type.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesType = reportType === "all" || report.type === reportType;
    
    return matchesSearch && matchesType;
  });

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Page Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">Reports</h1>
            <p className="text-muted-foreground">Access and manage your reports</p>
          </div>
          
          <Button className="bg-gradient-to-r from-primary to-secondary text-black">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" x2="12" y1="15" y2="3"/>
            </svg>
            Generate New Report
          </Button>
        </div>

        <Tabs defaultValue="all" className="w-full">
          <TabsList className="mb-6 w-full md:w-auto">
            <TabsTrigger value="all">All Reports</TabsTrigger>
            <TabsTrigger value="saved">Saved Reports</TabsTrigger>
            <TabsTrigger value="templates">Report Templates</TabsTrigger>
          </TabsList>
          
          <TabsContent value="all">
            {/* Filters */}
            <div className="flex flex-col md:flex-row gap-4 mb-6">
              <div className="w-full md:w-1/3">
                <Input 
                  placeholder="Search reports..." 
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full"
                />
              </div>
              <div className="w-full md:w-1/4">
                <Select value={reportType} onValueChange={setReportType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Filter by type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="statistical">Statistical</SelectItem>
                    <SelectItem value="analytical">Analytical</SelectItem>
                    <SelectItem value="educational">Educational</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="w-full md:w-1/4">
                <Select defaultValue="newest">
                  <SelectTrigger>
                    <SelectValue placeholder="Sort by" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="newest">Newest First</SelectItem>
                    <SelectItem value="oldest">Oldest First</SelectItem>
                    <SelectItem value="az">A-Z</SelectItem>
                    <SelectItem value="za">Z-A</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Reports Table */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle>Your Reports</CardTitle>
                <CardDescription>
                  Reports generated from your analyses and system data
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[120px]">Date</TableHead>
                      <TableHead>Title</TableHead>
                      <TableHead className="w-[120px]">Type</TableHead>
                      <TableHead className="w-[120px]">Status</TableHead>
                      <TableHead className="w-[120px] text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredReports.length > 0 ? (
                      filteredReports.map((report) => (
                        <TableRow key={report.id} className="hover:bg-muted/30 transition-colors">
                          <TableCell className="font-mono text-xs">
                            {report.date}
                          </TableCell>
                          <TableCell className="font-medium">{report.title}</TableCell>
                          <TableCell>{getTypeBadge(report.type)}</TableCell>
                          <TableCell>
                            <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">
                              {report.status}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-2">
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"/>
                                  <path d="M12 16v-4"/>
                                  <path d="M12 8h.01"/>
                                </svg>
                              </Button>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 0 0 2-2v-4"/>
                                  <path d="M17 9l-5 5-5-5"/>
                                  <path d="M12 12.8V2.5"/>
                                </svg>
                              </Button>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                                  <polyline points="17 21 17 13 7 13 7 21"/>
                                  <polyline points="7 3 7 8 15 8"/>
                                </svg>
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                          No reports found matching your filters
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="saved">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle>Saved Reports</CardTitle>
                <CardDescription>
                  Reports you've bookmarked for easy reference
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[120px]">Date</TableHead>
                      <TableHead>Title</TableHead>
                      <TableHead className="w-[120px]">Type</TableHead>
                      <TableHead className="w-[120px] text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {savedReports.map((report) => (
                      <TableRow key={report.id} className="hover:bg-muted/30 transition-colors">
                        <TableCell className="font-mono text-xs">
                          {report.date}
                        </TableCell>
                        <TableCell className="font-medium">{report.title}</TableCell>
                        <TableCell>{getTypeBadge(report.type)}</TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"/>
                                <path d="M12 16v-4"/>
                                <path d="M12 8h.01"/>
                              </svg>
                            </Button>
                            <Button variant="ghost" size="icon" className="h-8 w-8 text-yellow-500">
                              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                              </svg>
                            </Button>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                                <polyline points="17 21 17 13 7 13 7 21"/>
                                <polyline points="7 3 7 8 15 8"/>
                              </svg>
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="templates">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="hover:border-primary transition-colors cursor-pointer">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Analysis Summary</CardTitle>
                  <CardDescription>
                    Generate a comprehensive summary of your video analysis results
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex justify-between items-center">
                    <Badge variant="outline">Standard Template</Badge>
                    <Button variant="outline" size="sm">
                      Use Template
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="hover:border-primary transition-colors cursor-pointer">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Monthly Statistics</CardTitle>
                  <CardDescription>
                    Aggregate statistical data about your deepfake detection activities
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex justify-between items-center">
                    <Badge variant="outline">Standard Template</Badge>
                    <Button variant="outline" size="sm">
                      Use Template
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="hover:border-primary transition-colors cursor-pointer">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Trends Report</CardTitle>
                  <CardDescription>
                    Analyze trends in deepfake technology and detection methods
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex justify-between items-center">
                    <Badge variant="outline">Premium Template</Badge>
                    <Button variant="outline" size="sm">
                      Use Template
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="hover:border-primary transition-colors cursor-pointer">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Educational Brief</CardTitle>
                  <CardDescription>
                    Create an educational brief about deepfake detection techniques
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex justify-between items-center">
                    <Badge variant="outline">Standard Template</Badge>
                    <Button variant="outline" size="sm">
                      Use Template
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="hover:border-primary transition-colors cursor-pointer border-dashed border-2 flex flex-col items-center justify-center p-6">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mb-4 text-muted-foreground">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" x2="12" y1="8" y2="16"/>
                  <line x1="8" x2="16" y1="12" y2="12"/>
                </svg>
                <h3 className="text-lg font-medium mb-1">Create Custom Template</h3>
                <p className="text-sm text-muted-foreground text-center mb-4">
                  Design your own report template with custom fields and sections
                </p>
                <Button variant="outline">Create Template</Button>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}