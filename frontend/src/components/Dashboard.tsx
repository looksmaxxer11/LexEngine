import { useState, useEffect } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { useAuth } from "../contexts/AuthContext";
import { getUserHistory, deleteOCRResult, renameOCRResult, OCRResult } from "../lib/supabase";
import { FileText, Trash2, Download, Calendar, Clock, Pencil } from "lucide-react";

export function Dashboard() {
  const { user } = useAuth();
  const [history, setHistory] = useState<OCRResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  useEffect(() => {
    resetAndLoad();
  }, [user]);

  const resetAndLoad = async () => {
    if (!user) return;
    setLoading(true);
    setPage(1);
    const results = await getUserHistory(user.id, 1, 10);
    setHistory(results);
    setHasMore(results.length === 10);
    setLoading(false);
  };

  const loadMore = async () => {
    if (!user || !hasMore) return;
    const nextPage = page + 1;
    const more = await getUserHistory(user.id, nextPage, 10);
    setHistory(prev => [...prev, ...more]);
    setPage(nextPage);
    setHasMore(more.length === 10);
  };

  const handleDelete = async (id: string) => {
    if (!user) return;
    if (confirm("Are you sure you want to delete this result?")) {
      await deleteOCRResult(id, user.id);
      resetAndLoad();
    }
  };

  const handleRename = async (id: string, current: string) => {
    if (!user) return;
    const name = prompt("Rename document", current);
    if (!name || name.trim() === "" || name === current) return;
    const ok = await renameOCRResult(id, user.id, name.trim());
    if (ok) {
      setHistory(prev => prev.map(r => (r.id === id ? { ...r, filename: name.trim() } : r)));
    } else {
      alert("Failed to rename. Please try again.")
    }
  };

  const downloadText = (result: OCRResult) => {
    const blob = new Blob([result.text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${result.filename}_extracted.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full">
      {loading ? (
        <div className="flex flex-col items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <p className="text-slate-600 mt-4">Loading your history...</p>
        </div>
      ) : history.length === 0 ? (
          <div className="text-center py-16">
            <FileText className="w-16 h-16 text-slate-300 mx-auto mb-4" strokeWidth={1.5} />
            <h3 className="text-xl font-semibold text-slate-900 mb-2">No OCR Results Yet</h3>
            <p className="text-slate-600">Upload a document to get started!</p>
          </div>
        ) : (
          <>
          <div className="grid gap-4">
            {history.map((result) => (
              <Card key={result.id} className="p-6 hover:shadow-lg transition-shadow">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-3">
                      <FileText className="w-5 h-5 text-blue-600" />
                      <h3 className="text-lg font-semibold text-gray-900">
                        {result.filename}
                      </h3>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <FileText className="w-4 h-4" />
                        <span>{result.pages} pages</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <FileText className="w-4 h-4" />
                        <span>{result.columns} columns</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <FileText className="w-4 h-4" />
                        <span>{result.characters.toLocaleString()} chars</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <Clock className="w-4 h-4" />
                        <span>{result.processing_time}s</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Calendar className="w-4 h-4" />
                      <span>{new Date(result.created_at).toLocaleString()}</span>
                    </div>

                    <div className="mt-4 p-4 bg-gray-50 rounded-lg max-h-32 overflow-y-auto">
                      <p className="text-sm text-gray-700 line-clamp-3">
                        {result.text.substring(0, 200)}...
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2 ml-4">
                    <Button
                      onClick={() => handleRename(result.id, result.filename)}
                      variant="outline"
                      size="sm"
                    >
                      <Pencil className="w-4 h-4" />
                    </Button>
                    <Button
                      onClick={() => downloadText(result)}
                      variant="outline"
                      size="sm"
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                    <Button
                      onClick={() => handleDelete(result.id)}
                      variant="outline"
                      size="sm"
                      className="text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
          {hasMore && (
            <div className="mt-6 text-center">
              <Button variant="outline" onClick={loadMore}>Load More</Button>
            </div>
          )}
          </>
        )}
    </div>
  );
}
