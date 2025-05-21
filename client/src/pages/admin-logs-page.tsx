import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import { useToast } from "@/hooks/use-toast";
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
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function AdminLogsPage() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [logType, setLogType] = useState("all");
  
  // Mock logs data - would be replaced with actual API data
  const [logs, setLogs] = useState([
    {
      id: "log1",
      timestamp: "2025-05-21T14:32:18Z",
      type: "error",
      source: "api",
      message: "Failed to process video analysis request",
      details: "Error: 500 Internal Server Error"
    },
    {
      id: "log2",
      timestamp: "2025-05-21T13:56:42Z",
      type: "info",
      source: "auth",
      message: "User login successful",
      details: "User: admin"
    },
    {
      id: "log3",
      timestamp: "2025-05-21T13:45:11Z",
      type: "warning",
      source: "storage",
      message: "High storage usage detected",
      details: "Current usage: 85%"
    },
    {
      id: "log4",
      timestamp: "2025-05-21T12:32:09Z",
      type: "info",
      source: "system",
      message: "System update completed",
      details: "Version 2.3.0 installed successfully"
    },
    {
      id: "log5",
      timestamp: "2025-05-21T11:18:54Z",
      type: "error",
      source: "api",
      message: "API rate limit exceeded",
      details: "Too many requests from IP: 192.168.1.105"
    },
    {
      id: "log6",
      timestamp: "2025-05-21T10:42:33Z",
      type: "info",
      source: "auth",
      message: "New user registered",
      details: "Username: testuser"
    },
  ]);

  // Format timestamp for display
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  // Get badge color based on log type
  const getLogTypeBadge = (type: string) => {
    switch (type) {
      case "error":
        return <Badge variant="destructive">{type}</Badge>;
      case "warning":
        return <Badge variant="default" className="bg-yellow-500 hover:bg-yellow-600">{type}</Badge>;
      case "info":
        return <Badge variant="secondary">{type}</Badge>;
      default:
        return <Badge>{type}</Badge>;
    }
  };

  // Filter logs based on search term and type
  const filteredLogs = logs.filter(log => {
    const matchesSearch = 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.details.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.source.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesType = logType === "all" || log.type === logType;
    
    return matchesSearch && matchesType;
  });

  // Check if user is admin
  useEffect(() => {
    if (user && user.username !== "admin") {
      toast({
        title: "Access Denied",
        description: "You don't have permission to access this page.",
        variant: "destructive"
      });
      navigate("/dashboard");
    }
  }, [user, navigate, toast]);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar isAdmin />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Admin Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">System Logs</h1>
            <p className="text-muted-foreground">Monitor system activities and errors</p>
          </div>
          
          <div className="flex items-center gap-4">
            <Button variant="outline">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
              Export Logs
            </Button>
            <Button variant="destructive">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>
              Clear Logs
            </Button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="w-full md:w-1/3">
            <Input 
              placeholder="Search in logs..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full"
            />
          </div>
          <div className="w-full md:w-1/4">
            <Select value={logType} onValueChange={setLogType}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="error">Errors</SelectItem>
                <SelectItem value="warning">Warnings</SelectItem>
                <SelectItem value="info">Information</SelectItem>
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
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Logs Table */}
        <Card className="overflow-hidden">
          <CardHeader className="bg-muted/50 pb-3">
            <CardTitle>System Logs</CardTitle>
            <CardDescription>
              Showing {filteredLogs.length} of {logs.length} logs
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[180px]">Timestamp</TableHead>
                  <TableHead className="w-[100px]">Type</TableHead>
                  <TableHead className="w-[100px]">Source</TableHead>
                  <TableHead>Message</TableHead>
                  <TableHead className="w-[200px]">Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredLogs.length > 0 ? (
                  filteredLogs.map((log) => (
                    <TableRow key={log.id}>
                      <TableCell className="font-mono text-xs">
                        {formatTimestamp(log.timestamp)}
                      </TableCell>
                      <TableCell>{getLogTypeBadge(log.type)}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{log.source}</Badge>
                      </TableCell>
                      <TableCell>{log.message}</TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {log.details}
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                      No logs found matching your filters
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}