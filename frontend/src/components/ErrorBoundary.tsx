import { Component, ErrorInfo, ReactNode } from "react";
import { Button } from "./ui/button";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage?: string;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, errorMessage: error.message };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error in UI", error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, errorMessage: undefined });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4">
          <div className="max-w-lg w-full bg-white border border-slate-200 rounded-xl shadow-lg p-8 text-center space-y-4">
            <h1 className="text-2xl font-bold text-slate-900">Something went wrong</h1>
            <p className="text-slate-600">
              {this.state.errorMessage || "We're unable to render this view. Please try refreshing."}
            </p>
            <Button onClick={this.handleReset} className="bg-gradient-to-r from-blue-600 to-purple-600 text-white">
              Reload App
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
