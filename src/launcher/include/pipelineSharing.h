#ifndef PIPELINE_SHARING_H
#define PIPELINE_SHARING_H

#include <QObject>
#include <QString>
#include <QTimer>

class QProcess;
class QSettings;
class QWidget;
class PipelineShareSync;

// Sharing which shader and pipeline compilations a game needs with other players, so that they are
// compiled ahead on every computer (the server address is compiled in: KYTY_PIPELINE_SHARE_URL;
// without one none of this exists for the user).
namespace PipelineSharing {

// Whether a server address is configured: the checkbox of the settings and the consent dialog
// exist only then.
[[nodiscard]] bool Available();
// Whether the user agreed to share.
[[nodiscard]] bool Enabled();
void               SetEnabled(bool enabled);
// Asks the user once (the first time a game is launched). Returns whether sharing is on.
bool EnsureConsent(QWidget* parent);

void WriteSettings(QSettings& s);
void ReadSettings(QSettings& s);

} // namespace PipelineSharing

// The synchronization of one running game: a first round about 30 s after the launch, then one
// every 3 minutes, and a last one when the emulator ends.
class PipelineShareSession: public QObject {
	Q_OBJECT

public:
	PipelineShareSession(QProcess* process, const QString& title_id, const QString& emulator_dir,
	                     const QString& emulator_version, QObject* parent);
	~PipelineShareSession() override;

private:
	void Tick();
	void EmulatorEnded();
	void StartRound();

	QProcess*          m_process = nullptr;
	PipelineShareSync* m_sync    = nullptr;
	QTimer             m_timer;
	QTimer             m_watch;
	QString            m_title;
	bool               m_seen_emulator = false;
	bool               m_ended         = false;
	bool               m_final_started = false;
};

#endif // PIPELINE_SHARING_H
