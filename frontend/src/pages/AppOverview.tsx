import { useAuth } from "../contexts/AuthContext";
import { Card } from "../components/ui/card";
import { FileText, Upload as UploadIcon, Clock, CheckCircle } from "lucide-react";
import { Button } from "../components/ui/button";
import { Link } from "react-router-dom";
import { useState, useEffect } from "react";
import { getUserHistory, OCRResult } from "../lib/supabase";

export function AppOverview() {
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [recentActivity, setRecentActivity] = useState<OCRResult[]>([]);
  const [stats, setStats] = useState([
    { name: "Total Documents", value: "0", icon: FileText, color: "blue" },
    { name: "Processed Today", value: "0", icon: CheckCircle, color: "green" },
    { name: "Processing", value: "0", icon: Clock, color: "yellow" },
  ]);

  useEffect(() => {
    if (!user) return;
    
    const loadData = async () => {
      setLoading(true);
      try {
        // Fetch recent activity (last 5 documents)
        const results = await getUserHistory(user.id, 1, 5);
        setRecentActivity(results);
        
        // Calculate stats
        const allResults = await getUserHistory(user.id, 1, 1000); // Get all for stats
        const today = new Date().toDateString();
        const processedToday = allResults.filter(
          (r) => new Date(r.created_at).toDateString() === today
        ).length;
        
        setStats([
          { name: "Total Documents", value: allResults.length.toString(), icon: FileText, color: "blue" },
          { name: "Processed Today", value: processedToday.toString(), icon: CheckCircle, color: "green" },
          { name: "Processing", value: "0", icon: Clock, color: "yellow" },
        ]);
      } catch (error) {
        console.error("Error loading dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [user]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    
    if (hours < 1) return "Just now";
    if (hours < 24) return `${hours}h ago`;
    if (hours < 48) return "Yesterday";
    return date.toLocaleDateString();
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 break-words">
          Welcome back, {user?.email?.split("@")[0] || "User"}!
        </h1>
        <p className="text-slate-600 mt-2">
          Here's an overview of your OCR activity
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {stats.map((stat) => (
          <Card key={stat.name} className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm text-slate-600 mb-1">{stat.name}</p>
                {loading ? (
                  <div className="h-9 w-16 bg-slate-200 animate-pulse rounded"></div>
                ) : (
                  <p className="text-3xl font-bold text-slate-900">{stat.value}</p>
                )}
              </div>
              <div
                className={`
                  p-3 rounded-lg flex-shrink-0
                  ${stat.color === "blue" ? "bg-blue-100" : ""}
                  ${stat.color === "green" ? "bg-green-100" : ""}
                  ${stat.color === "yellow" ? "bg-yellow-100" : ""}
                `}
                aria-label={stat.name}
              >
                <stat.icon
                  className={`h-6 w-6
                    ${stat.color === "blue" ? "text-blue-600" : ""}
                    ${stat.color === "green" ? "text-green-600" : ""}
                    ${stat.color === "yellow" ? "text-yellow-600" : ""}
                  `}
                  aria-hidden="true"
                />
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Quick Actions */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-slate-900 mb-4">
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Link to="/app/upload">
            <Button
              size="lg"
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
            >
              <UploadIcon className="h-5 w-5 mr-2" />
              Upload New Document
            </Button>
          </Link>
          <Link to="/app/history">
            <Button size="lg" variant="outline" className="w-full">
              <FileText className="h-5 w-5 mr-2" />
              View History
            </Button>
          </Link>
        </div>
      </Card>

      {/* Recent Activity */}
      <Card className="p-6 mt-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-slate-900">
            Recent Activity
          </h2>
          {recentActivity.length > 0 && (
            <Link
              to="/app/history"
              className="text-sm text-blue-600 hover:text-blue-700 font-medium"
            >
              View All
            </Link>
          )}
        </div>
        
        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center gap-4 p-4 border border-slate-200 rounded-lg">
                <div className="h-10 w-10 bg-slate-200 animate-pulse rounded"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-slate-200 animate-pulse rounded w-1/3"></div>
                  <div className="h-3 bg-slate-200 animate-pulse rounded w-1/2"></div>
                </div>
              </div>
            ))}
          </div>
        ) : recentActivity.length === 0 ? (
          <div className="text-center py-12 text-slate-500">
            <FileText className="h-12 w-12 mx-auto mb-3 text-slate-300" />
            <p className="font-medium">No recent activity yet</p>
            <p className="text-sm mt-1">
              Upload your first document to get started
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {recentActivity.map((item) => (
              <Link
                key={item.id}
                to={`/app/history`}
                className="flex items-center gap-4 p-4 border border-slate-200 rounded-lg hover:border-blue-300 hover:bg-blue-50/50 transition-colors group"
              >
                <div className="p-2 bg-blue-100 rounded group-hover:bg-blue-200 transition-colors">
                  <FileText className="h-6 w-6 text-blue-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-900 truncate">
                    {item.filename}
                  </p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {formatDate(item.created_at)}
                  </p>
                </div>
                <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0" />
              </Link>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
