import { Building2, Receipt, FileStack, Briefcase } from "lucide-react";
import { Card } from "./ui/card";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import { Badge } from "./ui/badge";

const useCases = [
  {
    icon: Building2,
    title: "Enterprise Document Management",
    description: "Digitize and organize millions of documents for easy search and retrieval.",
    image: "https://images.unsplash.com/photo-1760952851538-17a59f691efe?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxidXNpbmVzcyUyMGF1dG9tYXRpb24lMjB3b3JrZmxvd3xlbnwxfHx8fDE3NjMwMjkzOTN8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    stat: "500K+",
    statLabel: "Docs/Day"
  },
  {
    icon: Receipt,
    title: "Invoice & Receipt Processing",
    description: "Automate expense management by extracting data from invoices and receipts.",
    image: "https://images.unsplash.com/photo-1600267165477-6d4cc741b379?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkaWdpdGFsJTIwZG9jdW1lbnRzfGVufDF8fHx8MTc2MzAyMjk0MHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    stat: "95%",
    statLabel: "Time Saved"
  },
  {
    icon: FileStack,
    title: "Legal Document Analysis",
    description: "Extract and analyze information from contracts and legal documents.",
    image: "https://images.unsplash.com/photo-1658142034594-12b5c411069d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkb2N1bWVudCUyMHNjYW5uaW5nJTIwdGVjaG5vbG9neXxlbnwxfHx8fDE3NjMwMzc4MTN8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    stat: "99.9%",
    statLabel: "Accuracy"
  },
  {
    icon: Briefcase,
    title: "HR & Onboarding",
    description: "Streamline employee onboarding by digitizing forms and identification documents.",
    image: "https://images.unsplash.com/photo-1600267165477-6d4cc741b379?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkaWdpdGFsJTIwZG9jdW1lbnRzfGVufDF8fHx8MTc2MzAyMjk0MHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    stat: "80%",
    statLabel: "Faster Onboarding"
  }
];

export function UseCases() {
  return (
    <section id="use-cases" className="py-24 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="container mx-auto">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-block bg-green-100 text-green-700 px-4 py-1 rounded-full mb-4">
            Use Cases
          </div>
          <h2 className="text-4xl sm:text-5xl mb-4">
            Built for Every Industry
          </h2>
          <p className="text-xl text-slate-600">
            From startups to Fortune 500 companies, our OCR solution adapts to your needs
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {useCases.map((useCase, index) => {
            const Icon = useCase.icon;
            return (
              <Card key={index} className="overflow-hidden bg-white border-slate-200 hover:shadow-2xl transition-all duration-300 group hover:-translate-y-1">
                <div className="relative h-56 overflow-hidden">
                  <ImageWithFallback 
                    src={useCase.image}
                    alt={useCase.title}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                  <Badge className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm text-slate-900 border-0">
                    <span className="text-lg">{useCase.stat}</span>
                    <span className="text-xs ml-1">{useCase.statLabel}</span>
                  </Badge>
                </div>
                <div className="p-6">
                  <div className="bg-gradient-to-br from-blue-600 to-purple-600 w-12 h-12 rounded-xl flex items-center justify-center mb-4 shadow-lg">
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <h3 className="mb-2">{useCase.title}</h3>
                  <p className="text-slate-600">{useCase.description}</p>
                </div>
              </Card>
            );
          })}
        </div>
      </div>
    </section>
  );
}