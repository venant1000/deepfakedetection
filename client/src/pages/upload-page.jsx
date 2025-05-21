import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { Loader2, Upload, CheckCircle2, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function UploadPage() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState("idle");

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith("video/")) {
        setFile(droppedFile);
      } else {
        toast({
          title: "Invalid file type",
          description: "Please upload a video file",
          variant: "destructive",
        });
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type.startsWith("video/")) {
        setFile(selectedFile);
      } else {
        toast({
          title: "Invalid file type",
          description: "Please upload a video file",
          variant: "destructive",
        });
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setIsUploading(true);
    setUploadStatus("uploading");
    
    // Simulate progress for better UX
    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => {
        const newProgress = prev + 5;
        if (newProgress >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return newProgress;
      });
    }, 200);

    try {
      const formData = new FormData();
      formData.append("video", file);

      const response = await fetch("/api/videos/upload", {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      clearInterval(progressInterval);
      
      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();
      setUploadProgress(100);
      setUploadStatus("success");
      
      toast({
        title: "Upload successful",
        description: "Your video has been uploaded and is being analyzed",
      });
      
      // Redirect to analysis page after 1.5 seconds
      setTimeout(() => {
        navigate(`/analysis/${data.id}`);
      }, 1500);
      
    } catch (error) {
      clearInterval(progressInterval);
      setUploadStatus("error");
      setUploadProgress(0);
      
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Please try again",
        variant: "destructive",
      });
      
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar isAdmin={user?.role === "admin"} />
      
      <div className="flex-1 ml-20 md:ml-64 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-8">Upload Video</h1>
          
          <div className="glass p-8 rounded-xl">
            <h2 className="text-xl font-semibold mb-4">Upload a video for deepfake analysis</h2>
            <p className="text-muted-foreground mb-6">
              Our AI will analyze the video and detect potential signs of manipulation.
              Supported formats: MP4, MOV, AVI, WEBM.
            </p>
            
            {uploadStatus === "idle" && (
              <>
                <div 
                  className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors mb-6 ${
                    isDragging 
                      ? "border-primary bg-primary/5" 
                      : "border-muted hover:border-primary/50 hover:bg-muted/50"
                  }`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById("file-upload")?.click()}
                >
                  <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept="video/*"
                    onChange={handleFileChange}
                  />
                  <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-medium mb-2">
                    Drag & Drop or Click to Upload
                  </h3>
                  <p className="text-muted-foreground text-sm">
                    Maximum file size: 100MB
                  </p>
                </div>
                
                {file && (
                  <div className="bg-muted/30 p-4 rounded-lg flex items-center justify-between mb-6">
                    <div className="flex items-center">
                      <div className="bg-primary/10 p-2 rounded-md mr-3">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="m10 8 6 4-6 4Z"/></svg>
                      </div>
                      <div>
                        <p className="font-medium">{file.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    <button 
                      className="text-muted-foreground hover:text-destructive"
                      onClick={() => setFile(null)}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>
                    </button>
                  </div>
                )}
                
                <Button 
                  onClick={handleUpload} 
                  className="w-full py-6"
                  disabled={!file || isUploading}
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    "Upload and Analyze"
                  )}
                </Button>
              </>
            )}
            
            {uploadStatus === "uploading" && (
              <div className="text-center py-8">
                <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
                <h3 className="text-lg font-medium mb-2">Uploading Video...</h3>
                <p className="text-muted-foreground mb-6">
                  Please don't close this window
                </p>
                
                <div className="h-2 bg-muted rounded-full mb-2 overflow-hidden">
                  <div 
                    className="h-full bg-primary" 
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-xs text-muted-foreground">
                  {uploadProgress}% Complete
                </p>
              </div>
            )}
            
            {uploadStatus === "success" && (
              <div className="text-center py-8">
                <CheckCircle2 className="h-12 w-12 mx-auto mb-4 text-green-500" />
                <h3 className="text-lg font-medium mb-2">Upload Successful!</h3>
                <p className="text-muted-foreground mb-6">
                  Redirecting to analysis...
                </p>
              </div>
            )}
            
            {uploadStatus === "error" && (
              <div className="text-center py-8">
                <AlertCircle className="h-12 w-12 mx-auto mb-4 text-destructive" />
                <h3 className="text-lg font-medium mb-2">Upload Failed</h3>
                <p className="text-muted-foreground mb-6">
                  There was an error uploading your video. Please try again.
                </p>
                <Button 
                  onClick={() => setUploadStatus("idle")}
                  variant="outline"
                >
                  Try Again
                </Button>
              </div>
            )}
          </div>
          
          <div className="mt-8 glass p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2 text-primary"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
              Tips for Best Results
            </h3>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary mt-1"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                Keep videos under 2 minutes for faster analysis
              </li>
              <li className="flex items-start gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary mt-1"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                Higher resolution videos provide more accurate results
              </li>
              <li className="flex items-start gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary mt-1"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                Good lighting and clear faces improve detection accuracy
              </li>
              <li className="flex items-start gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary mt-1"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                Avoid heavily compressed or low-quality videos
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}