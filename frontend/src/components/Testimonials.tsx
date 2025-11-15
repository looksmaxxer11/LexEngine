import { Star, Quote } from "lucide-react";
import { Card } from "./ui/card";
import { Avatar, AvatarFallback } from "./ui/avatar";

const testimonials = [
  {
    name: "Sarah Johnson",
    role: "CTO at TechCorp",
    company: "TechCorp",
    avatar: "SJ",
    rating: 5,
    text: "OCRify has transformed our document processing workflow. We've reduced processing time by 80% and the accuracy is phenomenal. The API integration was seamless.",
    gradient: "from-blue-400 to-blue-600"
  },
  {
    name: "Michael Chen",
    role: "Operations Director",
    company: "Global Logistics Inc",
    avatar: "MC",
    rating: 5,
    text: "The multi-language support is a game-changer for our international operations. We process invoices from 30+ countries with consistent accuracy.",
    gradient: "from-purple-400 to-purple-600"
  },
  {
    name: "Emily Rodriguez",
    role: "VP of Engineering",
    company: "FinanceHub",
    avatar: "ER",
    rating: 5,
    text: "Security and compliance were our top concerns. OCRify exceeded our expectations with SOC 2 and HIPAA compliance. Their support team is exceptional.",
    gradient: "from-pink-400 to-pink-600"
  },
  {
    name: "David Kim",
    role: "Product Manager",
    company: "DataFlow Systems",
    avatar: "DK",
    rating: 5,
    text: "The analytics dashboard gives us incredible insights into our document processing. We've optimized our entire workflow based on the data.",
    gradient: "from-cyan-400 to-cyan-600"
  },
  {
    name: "Lisa Anderson",
    role: "CEO",
    company: "LegalTech Solutions",
    avatar: "LA",
    rating: 5,
    text: "As a legal tech company, accuracy is everything. OCRify's 99.9% accuracy rate and ability to handle complex legal documents is unmatched.",
    gradient: "from-green-400 to-green-600"
  },
  {
    name: "James Wilson",
    role: "IT Director",
    company: "Healthcare United",
    avatar: "JW",
    rating: 5,
    text: "The on-premise deployment option was perfect for our healthcare compliance needs. Implementation was smooth and the team was very supportive.",
    gradient: "from-orange-400 to-orange-600"
  }
];

export function Testimonials() {
  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-white to-slate-50">
      <div className="container mx-auto">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-block bg-green-100 text-green-700 px-4 py-1 rounded-full mb-4">
            Testimonials
          </div>
          <h2 className="text-4xl sm:text-5xl mb-4">
            Loved by Teams Worldwide
          </h2>
          <p className="text-xl text-slate-600">
            See what our customers have to say about their experience with OCRify
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {testimonials.map((testimonial, index) => (
            <Card 
              key={index} 
              className="p-6 bg-white border-slate-200 hover:shadow-xl transition-all duration-300 hover:-translate-y-1"
            >
              <div className="flex items-start gap-4 mb-4">
                <Avatar className={`bg-gradient-to-br ${testimonial.gradient} w-12 h-12`}>
                  <AvatarFallback className="text-white">
                    {testimonial.avatar}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1">
                  <h4 className="mb-0">{testimonial.name}</h4>
                  <p className="text-sm text-slate-600">{testimonial.role}</p>
                  <p className="text-xs text-slate-500">{testimonial.company}</p>
                </div>
              </div>

              <div className="flex gap-1 mb-3">
                {Array.from({ length: testimonial.rating }).map((_, i) => (
                  <Star key={i} className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                ))}
              </div>

              <div className="relative">
                <Quote className="absolute -top-2 -left-2 h-8 w-8 text-slate-200" />
                <p className="text-slate-600 relative z-10 pl-6">
                  {testimonial.text}
                </p>
              </div>
            </Card>
          ))}
        </div>

        <div className="mt-12 text-center">
          <div className="inline-flex items-center gap-2 bg-slate-100 px-6 py-3 rounded-full">
            <div className="flex -space-x-2">
              {testimonials.slice(0, 4).map((t, i) => (
                <div 
                  key={i}
                  className={`w-10 h-10 rounded-full bg-gradient-to-br ${t.gradient} border-2 border-white flex items-center justify-center text-white text-sm`}
                >
                  {t.avatar}
                </div>
              ))}
            </div>
            <span className="text-slate-700">
              Join 10,000+ satisfied customers
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
