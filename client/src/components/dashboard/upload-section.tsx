import { useState } from "react";
import { Button } from "@/components/ui/button";

export default function UploadSection() {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };
  
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };
  
  const handleFile = (file: File) => {
    // Check file type and size
    const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
    const maxSize = 100 * 1024 * 1024; // 100MB
    
    if (!validTypes.includes(file.type)) {
      alert('Please upload a valid video file (MP4, MOV, AVI)');
      return;
    }
    
    if (file.size > maxSize) {
      alert('File size exceeds 100MB limit');
      return;
    }
    
    setFile(file);
  };
  
  const handleUpload = () => {
    if (!file) return;
    
    setUploading(true);
    
    // Simulate an upload process
    setTimeout(() => {
      setUploading(false);
      setFile(null);
      // In a real app, we would upload to the server and navigate to results
    }, 2000);
  };
  
  return (
    <div className="glass rounded-xl p-6 mb-8">
      <h2 className="text-xl font-semibold mb-4">Upload Video for Analysis</h2>
      
      <div 
        className={`border-2 border-dashed rounded-xl p-8 text-center ${
          isDragging ? 'border-primary bg-primary/5' : 'border-muted'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {!file ? (
          <>
            <div className="mb-4">
              <div className="h-16 w-16 mx-auto rounded-full flex items-center justify-center bg-primary/20 text-primary">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
              </div>
            </div>
            <h3 className="text-lg font-medium mb-2">Drag and drop your video file</h3>
            <p className="text-muted-foreground mb-4">or click to browse your files</p>
            <p className="text-xs text-muted-foreground mb-4">Supported formats: MP4, MOV, AVI (Max: 100MB)</p>
            
            <Button
              className="py-2 px-6 rounded-lg bg-primary text-black font-medium hover:opacity-90 transition-all"
              onClick={() => document.getElementById('file-input')?.click()}
            >
              Select File
            </Button>
            <input
              id="file-input"
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo"
              className="hidden"
              onChange={handleFileInput}
            />
          </>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-center mb-2">
              <div className="h-12 w-12 rounded-full flex items-center justify-center bg-primary/20 text-primary mr-3">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 8-6-6H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M10 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/><path d="m22 16-5.23-5.23a1 1 0 0 0-1.41 0L12 14.12l-1.36-1.36a1 1 0 0 0-1.41 0L2 20"/></svg>
              </div>
              <div className="text-left">
                <p className="font-medium">{file.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            </div>
            
            {uploading ? (
              <>
                <div className="w-full bg-muted rounded-full h-2">
                  <div className="bg-primary h-2 rounded-full animate-pulse" style={{ width: "70%" }}></div>
                </div>
                <p className="text-sm text-muted-foreground">Uploading: 70%</p>
              </>
            ) : (
              <div className="flex gap-3 justify-center">
                <Button
                  className="py-2 px-6 rounded-lg bg-primary text-black font-medium hover:opacity-90 transition-all"
                  onClick={handleUpload}
                >
                  Analyze Now
                </Button>
                <Button
                  variant="outline"
                  className="py-2 px-6 rounded-lg border border-muted text-muted-foreground hover:text-white hover:bg-muted transition-all"
                  onClick={() => setFile(null)}
                >
                  Cancel
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
