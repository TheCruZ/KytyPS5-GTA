#include "pipelineSharing.h"

#include "kytyGitVersion.h"
#include "pipelineShareSync.h"

#include <QAbstractButton>
#include <QCoreApplication>
#include <QDebug>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QSettings>
#include <QUrl>
#include <QUuid>

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep

#include <tlhelp32.h>
#endif

// The server of this build; empty compiles the feature out of the user's sight.
#ifndef KYTY_PIPELINE_SHARE_URL
#define KYTY_PIPELINE_SHARE_URL ""
#endif

namespace {

constexpr char SETTINGS_GROUP[]    = "PipelineSharing";
constexpr char SETTINGS_DECISION[] = "decision"; // -1 not asked yet, 0 no, 1 yes
constexpr char SETTINGS_CLIENT[]   = "client_id";
constexpr char SETTINGS_URL[]      = "url"; // For tests: replaces the compiled address.

// How long the launch of the game waits for the first round at most: a slow or unreachable
// server must not delay the game more than this; what arrives later enters through the emulator's
// reload of the shared manifest.
constexpr int PREFETCH_MAX_MS = 8 * 1000;
constexpr int FIRST_ROUND_MS = 30 * 1000;
constexpr int PERIOD_MS      = 3 * 60 * 1000;
constexpr int WATCH_MS       = 5 * 1000;

int     g_decision = -1;
QString g_client_id;
QString g_url_override;

QUrl ServerUrl() {
	const auto text = g_url_override.isEmpty() ? QString::fromLatin1(KYTY_PIPELINE_SHARE_URL)
	                                           : g_url_override;
	QUrl url(text.trimmed());
	if (!url.isValid() || url.host().isEmpty()) {
		return {};
	}
	// The data goes over TLS; plain HTTP only to the same computer (tests).
	const bool local = url.host() == QStringLiteral("localhost") ||
	                   url.host() == QStringLiteral("127.0.0.1") ||
	                   url.host() == QStringLiteral("::1");
	if (url.scheme() != QStringLiteral("https") && !(local && url.scheme() == QStringLiteral("http"))) {
		return {};
	}
	while (url.path().endsWith(QLatin1Char('/'))) {
		url.setPath(url.path().chopped(1));
	}
	return url;
}

QString ClientId() {
	if (g_client_id.isEmpty()) {
		g_client_id = QUuid::createUuid().toString(QUuid::WithoutBraces);
	}
	return g_client_id;
}

QString UserAgent() {
	return QStringLiteral("KytyPS5-GTA-launcher/") + QString::fromLatin1(KYTY_GIT_HASH);
}

#if defined(_WIN32)
// Whether a kyty_emulator.exe runs under the process (the launcher starts it through cmd).
bool EmulatorRunsUnder(DWORD parent) {
	const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		return false;
	}
	PROCESSENTRY32W entry {};
	entry.dwSize = sizeof(entry);
	bool found   = false;
	for (bool more = Process32FirstW(snapshot, &entry); more && !found;
	     more      = Process32NextW(snapshot, &entry)) {
		found = entry.th32ParentProcessID == parent &&
		        QString::fromWCharArray(entry.szExeFile).compare(QStringLiteral("kyty_emulator.exe"),
		                                                         Qt::CaseInsensitive) == 0;
	}
	CloseHandle(snapshot);
	return found;
}
#endif

} // namespace

namespace PipelineSharing {

bool Available() {
	return ServerUrl().isValid();
}

bool Enabled() {
	return Available() && g_decision == 1;
}

void SetEnabled(bool enabled) {
	g_decision = enabled ? 1 : 0;
	if (enabled) {
		ClientId();
	}
}

bool EnsureConsent(QWidget* parent) {
	if (!Available()) {
		return false;
	}
	if (g_decision >= 0) {
		return g_decision == 1;
	}
	QMessageBox box(QMessageBox::Question, QObject::tr("Share pipeline data?"),
	                QObject::tr("Games stutter the first time they need a shader or a pipeline, "
	                            "because your PC has to compile it. Other players have already "
	                            "compiled the ones this game uses, and sharing what each of us "
	                            "needs lets every PC compile them ahead.\n\n"
	                            "What is sent: the ID and version of the game, the version of the "
	                            "emulator, descriptions of the shaders and pipelines the game "
	                            "used (hashes and state flags; never game files or game code) and "
	                            "a random identifier of this installation. The server also sees "
	                            "your IP address.\n\n"
	                            "When: right before the game starts (the game never waits "
	                            "more than a few seconds for the server), about every 3 minutes "
	                            "while you play, and when the game ends.\n\n"
	                            "What for: the pipelines that other players already saw are "
	                            "compiled earlier on your PC, and the data helps us extend the "
	                            "manifests that come with the repository.\n\n"
	                            "You can change this at any time with the \"Share pipeline "
	                            "data\" checkbox at the bottom of the launcher window."),
	                QMessageBox::NoButton, parent);
	auto* share = box.addButton(QObject::tr("Share pipelines"), QMessageBox::AcceptRole);
	auto* no    = box.addButton(QObject::tr("Don't share"), QMessageBox::RejectRole);
	box.setDefaultButton(share);
	box.exec();
	// Closing the dialog decides nothing: it is asked again at the next launch.
	if (box.clickedButton() == static_cast<QAbstractButton*>(share)) {
		SetEnabled(true);
	} else if (box.clickedButton() == static_cast<QAbstractButton*>(no)) {
		SetEnabled(false);
	}
	return g_decision == 1;
}

void WriteSettings(QSettings& s) {
	s.beginGroup(SETTINGS_GROUP);
	s.setValue(SETTINGS_DECISION, g_decision);
	if (!g_client_id.isEmpty()) {
		s.setValue(SETTINGS_CLIENT, g_client_id);
	}
	s.endGroup();
}

void ReadSettings(QSettings& s) {
	s.beginGroup(SETTINGS_GROUP);
	g_decision = s.value(SETTINGS_DECISION, -1).toInt();
	if (g_decision < -1 || g_decision > 1) {
		g_decision = -1;
	}
	g_client_id    = s.value(SETTINGS_CLIENT).toString();
	g_url_override = s.value(SETTINGS_URL).toString();
	s.endGroup();
}

} // namespace PipelineSharing

PipelineShareSession::PipelineShareSession(QProcess* process, const QString& title_id,
                                           const QString& game_version,
                                           const QString& emulator_dir,
                                           const QString& emulator_version, QObject* parent)
    : QObject(parent), m_process(process), m_title(title_id) {
	PipelineShareSync::Options options;
	options.url              = ServerUrl();
	options.client_id        = ClientId();
	options.emulator_version = emulator_version;
	options.user_agent       = UserAgent();
	options.emulator_dir     = emulator_dir;
	options.title_id         = title_id;
	options.game_version     = game_version;
	m_sync                   = new PipelineShareSync(options, this);
	connect(m_sync, &PipelineShareSync::Finished, this, [this](bool ok, const QString& message) {
		// Silent for the user: one line in the log, and the next round tries again.
		qInfo().noquote() << message << (ok ? "" : "(will retry)");
		if (m_launch) {
			// The round before the launch: the game goes on, with or without what it brought.
			Launch();
		} else if (m_ended) {
			if (m_final_started) {
				deleteLater();
			} else {
				m_final_started = true;
				m_sync->Start();
			}
		}
	});
	m_timer.setSingleShot(true);
	connect(&m_timer, &QTimer::timeout, this, &PipelineShareSession::Tick);
	m_prefetch_timer.setSingleShot(true);
	connect(&m_prefetch_timer, &QTimer::timeout, this, [this] {
		qInfo() << "Pipeline sharing:" << m_title << "the server is slow, the game starts without waiting";
		Launch();
	});
	if (process != nullptr) {
		connect(process, &QProcess::stateChanged, this, [this](QProcess::ProcessState state) {
			if (state == QProcess::NotRunning) {
				EmulatorEnded();
			}
		});
#if defined(_WIN32)
		// cmd stays open after the emulator ends when the terminal is not closed on its own.
		connect(&m_watch, &QTimer::timeout, this, [this] {
			const auto pid = m_process != nullptr ? static_cast<DWORD>(m_process->processId()) : 0;
			if (pid == 0) {
				return;
			}
			if (EmulatorRunsUnder(pid)) {
				m_seen_emulator = true;
			} else if (m_seen_emulator) {
				EmulatorEnded();
			}
		});
#endif
	}
}

PipelineShareSession::~PipelineShareSession() = default;

void PipelineShareSession::LaunchAfterFetch(std::function<void()> launch) {
	m_launch = std::move(launch);
	m_prefetch_timer.start(PREFETCH_MAX_MS);
	m_sync->Start(true);
}

void PipelineShareSession::Launch() {
	m_prefetch_timer.stop();
	if (!m_launch) {
		return;
	}
	const auto launch = std::move(m_launch);
	m_launch          = nullptr;
	launch();
	StartSchedule();
}

void PipelineShareSession::StartSchedule() {
	if (m_ended) {
		return;
	}
	m_timer.start(FIRST_ROUND_MS);
#if defined(_WIN32)
	if (m_process != nullptr) {
		m_watch.start(WATCH_MS);
	}
#endif
}

void PipelineShareSession::StartRound() {
	if (!m_sync->IsRunning()) {
		m_sync->Start();
	}
}

void PipelineShareSession::Tick() {
	if (m_ended) {
		return;
	}
	StartRound();
	m_timer.start(PERIOD_MS);
}

void PipelineShareSession::EmulatorEnded() {
	if (m_ended) {
		return;
	}
	m_ended = true;
	m_timer.stop();
	m_watch.stop();
	// The last round, once the emulator has written its recording for the last time: a round in
	// progress read the files before that, so it is followed by one more.
	if (!m_sync->IsRunning()) {
		m_final_started = true;
		m_sync->Start();
	}
}
