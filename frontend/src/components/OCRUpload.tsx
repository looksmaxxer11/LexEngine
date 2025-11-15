import { useState } from "react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Progress } from "./ui/progress";
import { Upload, FileText, Download, Copy, RefreshCw, CheckCircle2 } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";

interface OCRStats {
  pages: number;
  columns: number;
  characters: number;
  time: number;
}

export function OCRUpload() {
  const { user } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<{ text: string; stats: OCRStats } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [usePhase2, setUsePhase2] = useState(false);
  const [usePhase3, setUsePhase3] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Idle");
  const [controller, setController] = useState<AbortController | null>(null);

  const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB limit

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    validateAndSetFile(droppedFile);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    validateAndSetFile(selectedFile ?? null);
  };

  const validateAndSetFile = (candidate: File | null) => {
    if (!candidate) {
      setError("Please choose a PDF file");
      return;
    }

    if (candidate.type !== "application/pdf") {
      setError("Unsupported file type. Upload a PDF document.");
      setFile(null);
      return;
    }

    if (candidate.size > MAX_FILE_SIZE_BYTES) {
      setError("File is too large. Maximum size is 25MB.");
      setFile(null);
      return;
    }

    setFile(candidate);
    setError(null);
  };

  const processDocument = async () => {
    if (!file) return;

    setProcessing(true);
    setProgress(0);
    setLogs([]);
    setResult(null);
    setError(null);
     setStatusMessage("Connecting to processing server...");

    const abortController = new AbortController();
    setController(abortController);

    const formData = new FormData();
    formData.append("file", file);
    if (usePhase2) formData.append("use_phase2", "true");
    if (usePhase3) formData.append("use_phase3", "true");
    if (user) formData.append("user_id", user.id);

    try {
      const response = await fetch("http://localhost:8000/api/process", {
        method: "POST",
        body: formData,
        signal: abortController.signal,
      });

      if (!response.ok || !response.body) {
        const errorMessage = await parseErrorResponse(response);
        throw new Error(errorMessage || "Processing service unavailable");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No response stream");

      setStatusMessage("Processing document...");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "progress") {
                setProgress(data.percent);
                setLogs((prev) => [...prev, data.message]);
                setStatusMessage(data.message);
              } else if (data.type === "complete") {
                setResult({ text: data.text, stats: data.stats });
                setProgress(100);
                setStatusMessage("Processing complete");
              } else if (data.type === "error") {
                throw new Error(data.message);
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("Processing cancelled");
        setStatusMessage("Upload cancelled");
      } else if (err instanceof Error) {
        setError(err.message);
        setStatusMessage("Processing failed");
      } else {
        setError("Processing failed");
        setStatusMessage("Processing failed");
      }
    } finally {
      setProcessing(false);
      setController(null);
    }
  };

  const parseErrorResponse = async (response: Response) => {
    try {
      const data = await response.clone().json();
      return data.detail || data.message;
    } catch (err) {
      try {
        return await response.text();
      } catch (_) {
        return null;
      }
    }
  };

  const cancelProcessing = () => {
    controller?.abort();
  };

  const downloadText = () => {
    if (!result) return;
    const blob = new Blob([result.text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${file?.name.replace(".pdf", "")}_extracted.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyText = () => {
    if (!result) return;
    navigator.clipboard.writeText(result.text);
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setProgress(0);
    setLogs([]);
    setError(null);
    setStatusMessage("Idle");
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
        <div className="mb-8">
          <p className="text-slate-600">
            Upload a PDF document and configure processing options below
          </p>
        </div>

        <Card className="p-6 shadow-lg border-slate-200">
          {!result ? (
            <>
              {/* Upload Zone */}
              <div
                role="button"
                tabIndex={0}
                aria-label="Upload PDF"
                aria-describedby="upload-help"
                className={`border-2 border-dashed rounded-xl p-16 text-center transition-all cursor-pointer ${
                  isDragging
                    ? "border-blue-500 bg-blue-50 scale-105"
                    : file
                    ? "border-green-400 bg-green-50"
                    : "border-slate-300 hover:border-blue-400 hover:bg-slate-50"
                }`}
                onDrop={handleDrop}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    const input = document.getElementById('file-input');
                    (input as HTMLInputElement | null)?.click();
                  }
                }}
              >
                {file ? (
                  <div className="space-y-4">
                    <FileText className="w-16 h-16 mx-auto text-green-600" />
                    <div>
                      <p className="text-xl font-semibold text-slate-900">{file.name}</p>
                      <p className="text-sm text-slate-600 mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation();
                        setFile(null);
                      }}
                      className="mt-2"
                    >
                      Remove File
                    </Button>
                  </div>
                ) : (
                  <>
                    <Upload className="w-20 h-20 mx-auto mb-6 text-slate-400" />
                    <p className="text-xl font-semibold text-slate-900 mb-2">
                      Drag and drop your PDF here
                    </p>
                    <p className="text-sm text-slate-600 mb-6">or</p>
                    <input
                      id="file-input"
                      type="file"
                      accept="application/pdf"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                    <Button 
                      variant="default"
                      size="lg"
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation();
                        document.getElementById('file-input')?.click();
                      }}
                      type="button"
                      className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                    >
                      Browse Files
                    </Button>
                    <p className="text-xs text-slate-500 mt-4">PDF files only, up to 25MB</p>
                  </>
                )}
              </div>
              <p id="upload-help" className="sr-only">Press Enter to open file picker. Only PDF up to 25MB.</p>

              {/* Processing Options */}
              <div className="mt-6 space-y-3">
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={usePhase2}
                    onChange={(e) => setUsePhase2(e.target.checked)}
                    className="w-5 h-5 rounded border-slate-300"
                  />
                  <div>
                    <span className="font-medium text-slate-900">Phase 2: Multi-Scale OCR</span>
                    <p className="text-sm text-slate-600">Enhanced accuracy with adaptive scaling</p>
                  </div>
                </label>

                <label className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={usePhase3}
                    onChange={(e) => setUsePhase3(e.target.checked)}
                    className="w-5 h-5 rounded border-slate-300"
                  />
                  <div>
                    <span className="font-medium text-slate-900">Phase 3: Post-OCR Correction</span>
                    <p className="text-sm text-slate-600">AI-powered text refinement</p>
                  </div>
                </label>
              </div>

              {/* Error Display */}
              {error && !processing && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-700 mb-3">{error}</p>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={processDocument} disabled={!file}>
                      <RefreshCw className="w-3 h-3 mr-2" />
                      Retry
                    </Button>
                    <Button variant="ghost" size="sm" onClick={reset}>
                      Reset
                    </Button>
                  </div>
                </div>
              )}

              {/* Process Button */}
              <Button
                onClick={processDocument}
                disabled={!file || processing}
                className="w-full mt-6 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-6 text-lg font-semibold"
              >
                {processing ? "Processing..." : "Process Document"}
              </Button>

              {/* Progress */}
              {processing && (
                <div className="mt-6 space-y-4">
                  <div className="flex items-center justify-between text-sm text-slate-600">
                    <span aria-live="polite">{statusMessage}</span>
                    <Button variant="outline" size="sm" onClick={cancelProcessing}>
                      Cancel
                    </Button>
                  </div>
                  <Progress value={progress} className="h-2" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress} />
                  <div className="bg-slate-50 rounded-lg p-4 max-h-48 overflow-y-auto" aria-live="polite">
                    {logs.length === 0 ? (
                      <p className="text-sm text-slate-500">Awaiting updates...</p>
                    ) : (
                      logs.map((log, idx) => (
                        <p key={idx} className="text-sm text-slate-700 py-1">
                          {log}
                        </p>
                      ))
                    )}
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              {/* Results */}
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
                    <CheckCircle2 className="w-6 h-6 text-green-600" />
                    Processing Complete
                  </h3>
                  <Button onClick={reset} variant="outline" size="sm">
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Process Another
                  </Button>
                </div>

                {/* Stats Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Card className="p-4 text-center bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
                    <div className="text-3xl font-bold text-blue-700">{result.stats.pages}</div>
                    <div className="text-sm text-blue-600 mt-1">Pages</div>
                  </Card>
                  <Card className="p-4 text-center bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
                    <div className="text-3xl font-bold text-purple-700">{result.stats.columns}</div>
                    <div className="text-sm text-purple-600 mt-1">Columns</div>
                  </Card>
                  <Card className="p-4 text-center bg-gradient-to-br from-green-50 to-green-100 border-green-200">
                    <div className="text-3xl font-bold text-green-700">
                      {result.stats.characters.toLocaleString()}
                    </div>
                    <div className="text-sm text-green-600 mt-1">Characters</div>
                  </Card>
                  <Card className="p-4 text-center bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200">
                    <div className="text-3xl font-bold text-orange-700">{result.stats.time}s</div>
                    <div className="text-sm text-orange-600 mt-1">Time</div>
                  </Card>
                </div>

                {/* Text Preview */}
                <div>
                  <h4 className="text-lg font-semibold text-slate-900 mb-3">Extracted Text</h4>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 max-h-96 overflow-y-auto">
                    <pre className="whitespace-pre-wrap text-sm text-slate-700 font-mono">
                      {result.text}
                    </pre>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3">
                  <Button onClick={downloadText} className="flex-1 bg-blue-600 hover:bg-blue-700">
                    <Download className="w-4 h-4 mr-2" />
                    Download Text
                  </Button>
                  <Button onClick={copyText} variant="outline" className="flex-1">
                    <Copy className="w-4 h-4 mr-2" />
                    Copy to Clipboard
                  </Button>
                </div>
              </div>
            </>
          )}
        </Card>
      </div>
    
  );
}
