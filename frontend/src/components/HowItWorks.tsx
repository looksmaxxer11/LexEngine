import { Upload, ScanSearch, Download, ArrowRight } from "lucide-react";
import { Card } from "./ui/card";

const steps = [
  {
    icon: Upload,
    title: "Upload Documents",
    description: "Drag and drop or upload your documents via API. Support for bulk uploads and multiple formats including PDF, images, and scanned documents.",
    color: "from-blue-500 to-cyan-500"
  },
  {
    icon: ScanSearch,
    title: "AI Processing",
    description: "Our advanced OCR engine extracts text, tables, and structured data automatically with 99.9% accuracy using machine learning.",
    color: "from-purple-500 to-pink-500"
  },
  {
    icon: Download,
    title: "Export Results",
    description: "Download or integrate extracted data in JSON, CSV, XML or your preferred format. Seamless API integration with webhooks.",
    color: "from-green-500 to-emerald-500"
  }
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-slate-50 to-white">
      <div className="container mx-auto">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-block bg-purple-100 text-purple-700 px-4 py-1 rounded-full mb-4">
            How It Works
          </div>
          <h2 className="text-4xl sm:text-5xl mb-4">
            Three Simple Steps to Automation
          </h2>
          <p className="text-xl text-slate-600">
            Get started in minutes with our intuitive platform
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto mb-12">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <div key={index} className="relative">
                <Card className="p-8 text-center bg-white border-slate-200 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 h-full">
                  <div className="relative inline-block mb-6">
                    <div className={`absolute inset-0 bg-gradient-to-br ${step.color} rounded-2xl blur-xl opacity-30`}></div>
                    <div className={`relative bg-gradient-to-br ${step.color} w-20 h-20 rounded-2xl flex items-center justify-center shadow-lg`}>
                      <Icon className="h-10 w-10 text-white" />
                    </div>
                    <div className="absolute -top-2 -right-2 bg-white rounded-full w-8 h-8 flex items-center justify-center border-2 border-blue-600 shadow-md">
                      <span className="text-sm">{index + 1}</span>
                    </div>
                  </div>
                  <h3 className="mb-3">{step.title}</h3>
                  <p className="text-slate-600">{step.description}</p>
                </Card>
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                    <ArrowRight className="h-8 w-8 text-blue-400" />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Demo section */}
        <div className="max-w-4xl mx-auto">
          <Card className="p-8 bg-gradient-to-br from-blue-50 to-purple-50 border-2 border-blue-100">
            <div className="text-center">
              <h3 className="text-2xl mb-4">See It In Action</h3>
              <p className="text-slate-600 mb-6">
                Watch how OCRify processes a document in real-time with incredible accuracy
              </p>
              <div className="bg-slate-900 rounded-xl p-8 shadow-2xl">
                <div className="aspect-video bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg flex items-center justify-center border border-slate-700">
                  <div className="text-center">
                    <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-white/10 flex items-center justify-center">
                      <ScanSearch className="h-10 w-10 text-blue-400" />
                    </div>
                    <p className="text-slate-400">Interactive Demo</p>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
}