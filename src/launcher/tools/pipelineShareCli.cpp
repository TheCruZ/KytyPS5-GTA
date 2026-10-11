// One round of the pipeline sharing from the command line: the same code as the launcher's.
//   pipeline_share_cli <url> <emulator folder> <title id> [client id]
#include "pipelineShareSync.h"

#include <QCoreApplication>
#include <QTextStream>
#include <QUuid>

int main(int argc, char* argv[]) {
	QCoreApplication app(argc, argv);
	QTextStream      out(stdout);
	if (argc < 4) {
		out << "usage: pipeline_share_cli <url> <emulator folder> <title id> [client id]\n";
		return 2;
	}
	PipelineShareSync::Options options;
	options.url              = QUrl(QString::fromLocal8Bit(argv[1]));
	options.emulator_dir     = QString::fromLocal8Bit(argv[2]);
	options.title_id         = QString::fromLocal8Bit(argv[3]);
	options.client_id        = argc > 4 ? QString::fromLocal8Bit(argv[4])
	                                    : QUuid::createUuid().toString(QUuid::WithoutBraces);
	options.emulator_version = QStringLiteral("pipeline_share_cli");
	options.user_agent       = QStringLiteral("KytyPS5-GTA-launcher/cli");
	int result               = 1;
	PipelineShareSync sync(options);
	QObject::connect(&sync, &PipelineShareSync::Finished, &app, [&](bool ok, const QString& text) {
		out << (ok ? "OK " : "FAILED ") << text << "\n";
		out.flush();
		result = ok ? 0 : 1;
		QCoreApplication::quit();
	});
	sync.Start();
	QCoreApplication::exec();
	return result;
}
