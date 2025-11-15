import { Dashboard } from "../components/Dashboard";

export function AppHistory() {
  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 break-words">
          Document History
        </h1>
        <p className="text-slate-600 mt-2">
          View and manage all your processed documents
        </p>
      </div>
      <Dashboard />
    </div>
  );
}
