import { Code2 } from "lucide-react";
import { Card } from "./ui/card";

const integrations = [
  { name: "Salesforce", logo: "https://img.logo.dev/salesforce.com?token=pk_X-1ZBsWGQ325ZntpFL-uVQ" },
  { name: "Google Drive", logo: "https://img.logo.dev/drive.google.com?token=pk_X-1ZBsWGQ325ZntpFL-uVQ" },
  { name: "Dropbox", logo: "https://img.logo.dev/dropbox.com?token=pk_X-1ZBsWGQ325ZntpFL-uVQ" },
  { name: "Microsoft", logo: "https://img.logo.dev/microsoft.com?token=pk_X-1ZBsWGQ325ZntpFL-uVQ" },
  { name: "Slack", logo: "https://img.logo.dev/slack.com?token=pk_X-1ZBsWGQ325ZntpFL-uVQ" },
  { name: "Zapier", logo: "https://img.logo.dev/zapier.com?token=pk_X-1ZBsWGQ325ZntpFL-uVQ" },
];

export function Integrations() {
  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="container mx-auto">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-block bg-purple-100 text-purple-700 px-4 py-1 rounded-full mb-4">
            Integrations
          </div>
          <h2 className="text-4xl sm:text-5xl mb-4">
            Integrate With Your Favorite Tools
          </h2>
          <p className="text-xl text-slate-600">
            Connect OCRify with the tools you already use. Our API makes integration seamless.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6 mb-12">
          {integrations.map((integration, index) => (
            <Card 
              key={index} 
              className="p-6 flex items-center justify-center bg-white border-slate-200 hover:shadow-lg hover:border-blue-200 transition-all duration-300 group cursor-pointer"
            >
              <div className="grayscale group-hover:grayscale-0 transition-all">
                <img 
                  src={integration.logo} 
                  alt={integration.name}
                  className="h-12 w-auto object-contain"
                />
              </div>
            </Card>
          ))}
        </div>

        <div className="max-w-4xl mx-auto">
          <Card className="p-8 bg-gradient-to-br from-slate-50 to-blue-50 border-2 border-blue-100">
            <div className="flex items-start gap-6">
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-4 rounded-xl shadow-lg">
                <Code2 className="h-8 w-8 text-white" />
              </div>
              <div className="flex-1">
                <h3 className="text-2xl mb-2">RESTful API</h3>
                <p className="text-slate-600 mb-4">
                  Integrate OCRify into your application with our comprehensive REST API. 
                  Complete documentation, SDKs for popular languages, and dedicated support.
                </p>
                <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
                  <code className="text-sm text-green-400">
                    <span className="text-blue-400">POST</span> https://api.ocrify.com/v1/process
                    <br />
                    <span className="text-slate-500">// Upload and process documents</span>
                  </code>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
}
